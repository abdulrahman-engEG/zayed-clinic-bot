import streamlit as st
import pandas as pd
import json
import re
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
import gspread

# ==========================================
# 1. إعدادات الهوية والبيزنس (White-Label Config)
# ==========================================
CLINIC_NAME = "عيادة زايد سمايل للأسنان"
CLINIC_ICON = "🦷"
CLINIC_ADDRESS = "أركان بلازا، المبنى 4، الدور الثاني، الشيخ زايد"
WHATSAPP_NUMBER = "201001234567"
PDF_FILE_NAME = "clinic_data.pdf"
SHEET_NAME = "Clinic_Bookings"
ADMIN_PASSWORD = "1234"

CLINIC_WORKING_HOURS = "من السبت إلى الخميس، من الساعة 1 ظهراً حتى 10 مساءً (مدة الجلسة ساعة واحدة: 1م، 2م، 3م، 4م، 5م، 6م، 7م، 8م، 9م). الجمعة إجازة رسمية."
EXPECTED_HEADERS = ["وقت التسجيل", "اسم العميل", "رقم الموبايل", "الخدمة", "اليوم", "التاريخ", "الساعة"]

st.set_page_config(page_title=f"نظام إدارة وحجز - {CLINIC_NAME}", page_icon=CLINIC_ICON, layout="wide")

st.markdown("""
<style>
    .stButton>button { border-radius: 20px; border: 1px solid #00a884; font-size: 14px; }
    .whatsapp-btn {
        display: inline-block; background-color: #25D366; color: white !important;
        padding: 10px 20px; border-radius: 25px; text-decoration: none;
        font-weight: bold; text-align: center; width: 100%; margin-top: 10px;
    }
    .ticket-container {
        display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px;
    }
    .ticket-btn-client {
        display: inline-block; background-color: #25D366; color: white !important;
        padding: 10px 18px; border-radius: 20px; text-decoration: none;
        font-weight: bold; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.12);
    }
    .ticket-btn-clinic {
        display: inline-block; background-color: #128C7E; color: white !important;
        padding: 10px 18px; border-radius: 20px; text-decoration: none;
        font-weight: bold; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# قراءة المفاتيح من Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
    google_creds_info = {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"].strip(),
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"].strip(),
        "refresh_token": st.secrets["GOOGLE_REFRESH_TOKEN"].strip(),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    tg_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat_id = str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
except Exception as e:
    st.error(f"برجاء التأكد من إضافة جميع المفاتيح في إعدادات Secrets. التفاصيل: {e}")
    st.stop()

# ==========================================
# 2. دوال التوقيت، تذاكر الواتساب، الإشعارات، و Google Sheets
# ==========================================
def get_cairo_now():
    """الحصول على التاريخ والوقت الفعلي في القاهرة (UTC+2)"""
    cairo_tz = timezone(timedelta(hours=2))
    return datetime.now(cairo_tz)

def format_egypt_phone(raw_phone):
    """تحويل رقم الموبايل المصري لصيغة واتساب الدولية (201xxxxxxxxx)"""
    digits = re.sub(r"\D", "", str(raw_phone))
    if digits.startswith("01") and len(digits) == 11:
        return "2" + digits
    elif digits.startswith("1") and len(digits) == 10:
        return "20" + digits
    elif digits.startswith("0020"):
        return digits[2:]
    return digits

def build_ticket_buttons(client_phone, ticket_message):
    """إنشاء زرارين: حفظ التذكرة للعميل + إرسالها لواتساب العيادة"""
    encoded_msg = urllib.parse.quote(ticket_message)
    clean_client_wa = format_egypt_phone(client_phone)
    
    client_btn = ""
    if clean_client_wa:
        client_btn = f'<a href="https://wa.me/{clean_client_wa}?text={encoded_msg}" target="_blank" class="ticket-btn-client">📲 احفظ التذكرة على واتسابك</a>'
    
    clinic_btn = f'<a href="https://wa.me/{WHATSAPP_NUMBER}?text={encoded_msg}" target="_blank" class="ticket-btn-clinic">📤 أرسل التذكرة لواتساب العيادة</a>'
    
    return f'\n\n<div class="ticket-container">{client_btn}{clinic_btn}</div>'

def send_telegram_alert(message_text):
    """إرسال إشعار فوري للإدارة عبر Telegram Bot API بدون مكتبات خارجية"""
    if not tg_token or not tg_chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": tg_chat_id,
            "text": message_text,
            "parse_mode": "HTML"
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False

ARABIC_DAYS = {
    "Monday": "الإثنين", "Tuesday": "الثلاثاء", "Wednesday": "الأربعاء",
    "Thursday": "الخميس", "Friday": "الجمعة", "Saturday": "السبت", "Sunday": "الأحد"
}

def get_calendar_context():
    now = get_cairo_now()
    current_time_str = now.strftime("%I:%M %p").replace("AM", "صباحاً").replace("PM", "مساءً")
    lines = [
        f"الوقت الحالي الآن: {ARABIC_DAYS[now.strftime('%A')]} {now.strftime('%Y-%m-%d')} الساعة {current_time_str}",
        "جدول الأيام القادمة:"
    ]
    for i in range(14):
        d = now + timedelta(days=i)
        day_ar = ARABIC_DAYS[d.strftime("%A")]
        label = " (اليوم)" if i == 0 else (" (غداً)" if i == 1 else "")
        lines.append(f"- {day_ar} الموافق {d.strftime('%Y-%m-%d')}{label}")
    return "\n".join(lines)

@st.cache_resource
def get_gsheet():
    creds = Credentials.from_authorized_user_info(
        google_creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    return sh.sheet1

def ensure_headers(ws):
    first_row = ws.row_values(1)
    if not first_row:
        ws.append_row(EXPECTED_HEADERS)

def fetch_all_bookings():
    try:
        ws = get_gsheet()
        ensure_headers(ws)
        records = ws.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"تعذر قراءة Google Sheet حالياً: {e}")
        return pd.DataFrame(columns=EXPECTED_HEADERS)

def add_booking_to_sheet(name, phone, service, day, date_str, time_slot):
    ws = get_gsheet()
    ensure_headers(ws)
    
    # فحص منع التكرار: التأكد أن الموعد غير مسجل مسبقاً لنفس العميل
    records = ws.get_all_records()
    clean_phone = re.sub(r"\D", "", str(phone))
    for row in records:
        row_phone = re.sub(r"\D", "", str(row.get("رقم الموبايل", "")))
        if (str(row.get("التاريخ", "")).strip() == str(date_str).strip() and 
            str(row.get("الساعة", "")).strip() == str(time_slot).strip() and 
            clean_phone and clean_phone in row_phone):
            return False  # الحجز مسجل بالفعل، لا تكرره

    now_str = get_cairo_now().strftime("%Y-%m-%d %H:%M")
    ws.append_row([now_str, name, str(phone), service, day, date_str, time_slot])
    
    # إرسال إشعار فوري للإدارة
    alert_msg = (
        f"🔔 <b>حجز جديد - {CLINIC_NAME}</b>\n\n"
        f"👤 <b>العميل:</b> {name}\n"
        f"📱 <b>الموبايل:</b> <code>{phone}</code>\n"
        f"🩺 <b>الخدمة:</b> {service}\n"
        f"📅 <b>الموعد:</b> {day} ({date_str})\n"
        f"⏰ <b>الساعة:</b> {time_slot}"
    )
    send_telegram_alert(alert_msg)
    return True

def delete_booking_by_phone(phone):
    ws = get_gsheet()
    ensure_headers(ws)
    records = ws.get_all_records()
    clean_target = re.sub(r"\D", "", str(phone))
    deleted = False
    client_name = ""
    for idx in range(len(records) - 1, -1, -1):
        row_phone = re.sub(r"\D", "", str(records[idx].get("رقم الموبايل", "")))
        if clean_target and clean_target in row_phone:
            client_name = records[idx].get("اسم العميل", "عميل")
            ws.delete_rows(idx + 2)
            deleted = True
            break
    if deleted:
        alert_msg = (
            f"❌ <b>إلغاء حجز - {CLINIC_NAME}</b>\n\n"
            f"👤 <b>العميل:</b> {client_name}\n"
            f"📱 <b>الموبايل:</b> <code>{phone}</code>\n"
            f"ℹ️ تم تحرير الموعد في الجدول تلقائياً."
        )
        send_telegram_alert(alert_msg)
    return deleted

def update_booking_by_phone(phone, new_day, new_date, new_time):
    ws = get_gsheet()
    ensure_headers(ws)
    records = ws.get_all_records()
    clean_target = re.sub(r"\D", "", str(phone))
    updated = False
    client_name = ""
    for idx in range(len(records) - 1, -1, -1):
        row_phone = re.sub(r"\D", "", str(records[idx].get("رقم الموبايل", "")))
        if clean_target and clean_target in row_phone:
            client_name = records[idx].get("اسم العميل", "عميل")
            row_num = idx + 2
            ws.update_cell(row_num, 5, new_day)
            ws.update_cell(row_num, 6, new_date)
            ws.update_cell(row_num, 7, new_time)
            updated = True
            break
    if updated:
        alert_msg = (
            f"🔄 <b>تعديل موعد حجز - {CLINIC_NAME}</b>\n\n"
            f"👤 <b>العميل:</b> {client_name}\n"
            f"📱 <b>الموبايل:</b> <code>{phone}</code>\n"
            f"📅 <b>الموعد الجديد:</b> {new_day} ({new_date}) الساعة {new_time}"
        )
        send_telegram_alert(alert_msg)
    return updated

def delete_booking_row(row_index_in_df):
    ws = get_gsheet()
    ws.delete_rows(int(row_index_in_df) + 2)

# ==========================================
# 3. تجهيز قاعدة المعرفة (RAG)
# ==========================================
@st.cache_resource
def load_knowledge_base():
    pdf_reader = PdfReader(PDF_FILE_NAME)
    raw_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

with st.spinner("جاري تحميل النظام الذكي وربط قاعدة البيانات..."):
    vector_store = load_knowledge_base()

# ==========================================
# 4. تقسيم الواجهة: واجهة العملاء + لوحة الإدارة
# ==========================================
tab_chat, tab_admin = st.tabs(["💬 مساعد العملاء والحجز الذكي", "📊 لوحة تحكم الإدارة والتحليلات"])

# ------------------------------------------
# TAB 1: محادثة العملاء (حجز / تعديل / إلغاء / تذاكر واتساب)
# ------------------------------------------
with tab_chat:
    col_header, col_wa = st.columns([3, 1])
    with col_header:
        st.title(f"{CLINIC_ICON} {CLINIC_NAME}")
        st.caption("اسأل عن الأسعار، احجز ميعادك بالتاريخ الدقيق، أو عدّل/ألغِ حجزك في ثوانٍ.")
    with col_wa:
        wa_link = f"https://wa.me/{WHATSAPP_NUMBER}?text=أهلاً،%20بتواصل%20معاكم%20من%20المساعد%20الذكي"
        st.markdown(f'<a href="{wa_link}" target="_blank" class="whatsapp-btn">💬 واتساب مباشر</a>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"أهلاً بحضرتك في {CLINIC_NAME}! 👋 إزاي أقدر أساعدك النهاردة؟ تقدر تسأل عن الأسعار، تحجز ميعاد جديد، أو تعدل/تلغي ميعادك الحالي."}
        ]

    q_cols = st.columns(4)
    quick_question = None
    if q_cols[0].button("💰 قائمة الأسعار"): quick_question = "إيه قائمة الأسعار عندكم؟"
    if q_cols[1].button("📅 حجز ميعاد جديد"): quick_question = "عايز أحجز ميعاد جديد"
    if q_cols[2].button("🔄 تعديل ميعادي"): quick_question = "عايز أعدل ميعاد حجزي"
    if q_cols[3].button("❌ إلغاء حجز"): quick_question = "عايز ألغي حجزي"

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    typed_question = st.chat_input("اكتب رسالتك هنا (حجز، استفسار، تعديل أو إلغاء)...")
    user_question = quick_question if quick_question else typed_question

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        df_current = fetch_all_bookings()
        if not df_current.empty and "التاريخ" in df_current.columns and "الساعة" in df_current.columns:
            booked_list = [
                f"- {row.get('اليوم', '')} ({row.get('التاريخ', '')}) الساعة {row.get('الساعة', '')} [اسم: {row.get('اسم العميل', '')} | موبايل: {row.get('رقم الموبايل', '')}]"
                for _, row in df_current.iterrows()
            ]
            booked_slots_text = "\n".join(booked_list)
        else:
            booked_slots_text = "لا توجد مواعيد محجوزة حالياً، جميع الساعات متاحة."

        calendar_text = get_calendar_context()
        docs = vector_store.similarity_search(user_question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        recent_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-8:]])

        prompt = f"""
        أنت موظف استقبال ومنسق مواعيد ذكي جداً ولبق في {CLINIC_NAME}.
        تحدث بالعامية المصرية المهذبة والودودة.

        === الوقت الحالي والتقويم الفعلي للأسبوعين القادمين ===
        {calendar_text}

        === مواعيد عمل العيادة ===
        {CLINIC_WORKING_HOURS}

        === سجل المواعيد المحجوزة حالياً ===
        {booked_slots_text}

        === بيانات العيادة والأسعار ===
        {context}

        === سجل المحادثة ===
        {recent_history}

        === رسالة العميل الحالية ===
        {user_question}

        === قواعد الذكاء والتدقيق الصارمة ===
        1. **التحقق من رقم الموبايل المصري:** يجب أن يتكون رقم الموبايل من 11 رقماً ويبدأ بـ (010 أو 011 أو 012 أو 015). إذا كتب العميل رقماً ناقصاً أو زائداً، نبهه بلطف واطلب منه تصحيح الرقم قبل متابعة الحجز.
        2. **التحقق من الساعة ومواعيد العمل والوقت المنقضي:**
           - إذا ذكر العميل ساعة برقم فقط بدون تحديد (صباحاً أم مساءً/ظهراً) مثل "الساعة 2" أو "الساعة 9"، اسأله للتوضيح أولاً (هل تقصد الساعة 2 ظهراً أم صباحاً؟) ولا تفترض التوقيت من تلقاء نفسك.
           - إذا طلب العميل يوم الجمعة (إجازة) أو ساعة خارج مواعيد العمل الرسمية (من 1 ظهراً إلى 9 مساءً)، اعتذر بلطف، وضح مواعيد العمل، واقترح أقرب مواعيد متاحة.
           - إذا طلب العميل الحجز في نفس يومنا هذا في ساعة قد فاتت بالفعل مقارنة بـ "الوقت الحالي الآن"، نبهه بلطف أن هذا التوقيت قد مر واقترح عليه الساعات المتبقية اليوم أو غداً.
        3. **منع تضارب المواعيد والترشيح الذكي:** طابق التاريخ (`YYYY-MM-DD`) والساعة مع "سجل المواعيد المحجوزة حالياً". إذا كان محجوزاً، اعتذر بلطف واعرض 3 ساعات فارغة في نفس اليوم. وإذا اختار العميل اليوم فقط دون تحديد ساعة، رشح له 3 ساعات متاحة من الجدول مباشرة.
        4. **فهم الأعراض وترشيح الخدمة:** إذا وصف العميل ألماً أو شكوى (مثل ألم الضرس مع الساقع أو اصفرار الأسنان) دون تسمية الخدمة، رشح له الخدمة المناسبة وسعرها من بيانات العيادة واعرض عليه حجز موعد.
        5. **الحجز الجديد (CREATE):** اجمع (الاسم بالكامل، رقم الموبايل الصحيح، الخدمة، اليوم، التاريخ YYYY-MM-DD، والساعة محددة بـ ظهراً/مساءً). عندما تكتمل البيانات وتصبح صحيحة ومتاحة، اعرض ملخص الحجز والتكلفة واسأل العميل للتأكيد النهائي **بدون كتابة ACTION_JSON في هذه الرسالة إطلاقاً**. فقط بعد أن يرد العميل بالموافقة الصريحة (مثل: نعم، أكد، تمام)، أرسل رسالة التأكيد النهائية وأضف في نهايتها السطر التالي مرة واحدة فقط:
           ACTION_JSON: {{"action": "create", "name": "الاسم", "phone": "الموبايل", "service": "الخدمة", "day": "اليوم", "date": "YYYY-MM-DD", "time": "الساعة"}}
        6. **تعديل موعد قائم (UPDATE):** اطلب رقم موبايله وتأكد من وجود حجز له، ثم اسأله عن اليوم والتاريخ والساعة الجديدة (مع تطبيق نفس قواعد التدقيق). عند تأكيد التعديل النهائي، أضف في نهاية ردك:
           ACTION_JSON: {{"action": "update", "phone": "الموبايل", "day": "اليوم_الجديد", "date": "YYYY-MM-DD", "time": "الساعة_الجديدة"}}
        7. **إلغاء موعد قائم (CANCEL):** اطلب رقم موبايله وتأكد من وجود حجز له. عند تأكيد الإلغاء النهائي، أضف في نهاية ردك:
           ACTION_JSON: {{"action": "cancel", "phone": "الموبايل"}}
        """

        with st.chat_message("assistant"):
            with st.spinner("جاري التنسيق مع جدول المواعيد..."):
                reply_text = None
                last_err = ""
                
                client = genai.Client(
                    api_key=api_key,
                    http_options={'api_version': 'v1beta'}
                )
                
                try:
                    available_models = [
                        m.name.replace("models/", "")
                        for m in client.models.list()
                        if "generateContent" in (m.supported_actions or []) and "gemini" in m.name.lower()
                    ]
                    flash_models = [m for m in available_models if "flash" in m.lower()]
                    models_to_try = flash_models if flash_models else available_models

                    for model_name in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(temperature=0.2),
                            )
                            reply_text = response.text
                            break
                        except Exception as e:
                            last_err = str(e)
                            continue
                except Exception as e:
                    last_err = str(e)

                if reply_text:
                    ticket_html = ""
                    action_match = re.search(r"ACTION_JSON:\s*(\{.*?\})", reply_text)
                    if action_match:
                        try:
                            act_data = json.loads(action_match.group(1))
                            act_type = act_data.get("action")
                            
                            if act_type == "create":
                                c_name = act_data.get("name", "غير محدد")
                                c_phone = act_data.get("phone", "")
                                c_service = act_data.get("service", "كشف")
                                c_day = act_data.get("day", "")
                                c_date = act_data.get("date", "")
                                c_time = act_data.get("time", "")

                                is_new = add_booking_to_sheet(c_name, c_phone, c_service, c_day, c_date, c_time)
                                if is_new:
                                    st.toast("✅ تم تسجيل الميعاد وإرسال إشعار للإدارة!", icon="📅")

                                ticket_msg = (
                                    f"🎫 *تذكرة حجز مؤكدة - {CLINIC_NAME}*\n"
                                    f"👤 الاسم: {c_name}\n"
                                    f"🩺 الخدمة: {c_service}\n"
                                    f"📅 الموعد: {c_day} ({c_date})\n"
                                    f"⏰ الساعة: {c_time}\n"
                                    f"📍 العنوان: {CLINIC_ADDRESS}"
                                )
                                ticket_html = build_ticket_buttons(c_phone, ticket_msg)
                                
                            elif act_type == "update":
                                u_phone = act_data.get("phone", "")
                                u_day = act_data.get("day", "")
                                u_date = act_data.get("date", "")
                                u_time = act_data.get("time", "")

                                ok = update_booking_by_phone(u_phone, u_day, u_date, u_time)
                                if ok:
                                    st.toast("🔄 تم تعديل ميعادك وإبلاغ الإدارة!", icon="✅")
                                    ticket_msg = (
                                        f"🔄 *تذكرة تعديل موعد - {CLINIC_NAME}*\n"
                                        f"📱 رقم الموبايل: {u_phone}\n"
                                        f"📅 الموعد الجديد: {u_day} ({u_date})\n"
                                        f"⏰ الساعة: {u_time}\n"
                                        f"📍 العنوان: {CLINIC_ADDRESS}"
                                    )
                                    ticket_html = build_ticket_buttons(u_phone, ticket_msg)
                                    
                            elif act_type == "cancel":
                                ok = delete_booking_by_phone(act_data.get("phone", ""))
                                if ok:
                                    st.toast("🗑️ تم إلغاء الحجز وإبلاغ الإدارة!", icon="ℹ️")
                                    
                        except Exception as sheet_err:
                            st.error(f"حدث خطأ أثناء تحديث قاعدة البيانات: {sheet_err}")
                        
                        reply_text = re.sub(r"ACTION_JSON:\s*\{.*?\}", "", reply_text).strip() + ticket_html

                    st.markdown(reply_text, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                else:
                    st.error(f"تعذر الرد من الموديل. التفاصيل: {last_err}")

# ------------------------------------------
# TAB 2: لوحة تحكم الإدارة والتحليلات الذكية
# ------------------------------------------
with tab_admin:
    st.header("📊 لوحة تحكم الإدارة والتحليلات (Live CRM)")
    admin_pass = st.text_input("أدخل كلمة مرور الإدارة:", type="password", placeholder="جرب: 1234")
    
    if admin_pass == ADMIN_PASSWORD:
        c_btn1, c_btn2 = st.columns([1, 1])
        with c_btn1:
            if st.button("🔄 تحديث البيانات الآن"):
                st.rerun()
        with c_btn2:
            if st.button("🔔 إرسال إشعار تجريبي لتيليجرام"):
                if send_telegram_alert(f"✅ تجربة اتصال ناجحة من نظام {CLINIC_NAME}!"):
                    st.success("تم إرسال الإشعار التجريبي على تيليجرام بنجاح!")
                else:
                    st.warning("تأكد من إضافة TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID في Secrets.")

        df_bookings = fetch_all_bookings()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("إجمالي الحجوزات النشطة", len(df_bookings))
        m2.metric("تاريخ اليوم", get_cairo_now().strftime("%Y-%m-%d"))
        m3.metric("حالة الإشعارات الفورية", "مفعّلة 🟢" if (tg_token and tg_chat_id) else "غير مفعّلة ⚪")
        
        st.divider()
        
        if not df_bookings.empty:
            st.subheader("📈 تحليلات الأداء والخدمات")
            c1, c2 = st.columns(2)
            with c1:
                if "الخدمة" in df_bookings.columns:
                    st.markdown("**توزيع الحجوزات حسب الخدمة:**")
                    st.bar_chart(df_bookings["الخدمة"].value_counts())
            with c2:
                if "اليوم" in df_bookings.columns:
                    st.markdown("**أكثر أيام الأسبوع طلباً:**")
                    st.bar_chart(df_bookings["اليوم"].value_counts())

            st.divider()
            st.subheader("📋 جدول المواعيد المؤكدة")
            st.dataframe(df_bookings, use_container_width=True)
            
            col_dl, col_del = st.columns([1, 2])
            with col_dl:
                csv = df_bookings.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 تحميل سجل المواعيد (Excel / CSV)",
                    data=csv,
                    file_name=f"bookings_{get_cairo_now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            with col_del:
                row_to_del = st.selectbox(
                    "حذف حجز يدوياً (اختر الصف):",
                    options=df_bookings.index,
                    format_func=lambda x: f"صف {x}: {df_bookings.loc[x, 'اسم العميل']} - {df_bookings.loc[x, 'التاريخ']} ({df_bookings.loc[x, 'الساعة']})"
                )
                if st.button("🗑️ حذف الميعاد المحدد"):
                    delete_booking_row(row_to_del)
                    st.success("تم حذف الميعاد بنجاح!")
                    st.rerun()
        else:
            st.info("لا توجد حجوزات مسجلة حالياً.")
    elif admin_pass:
        st.error("كلمة المرور غير صحيحة!")
