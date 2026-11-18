import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
import gspread

# ==========================================
# 1. إعدادات الهوية والبيزنس (تتغير بسهولة لأي عميل)
# ==========================================
CLINIC_NAME = "عيادة زايد سمايل للأسنان"
CLINIC_ICON = "🦷"
WHATSAPP_NUMBER = "201001234567"
PDF_FILE_NAME = "clinic_data.pdf"
SHEET_NAME = "Clinic_Bookings"
ADMIN_PASSWORD = "1234"

# مواعيد العمل الافتراضية لحساب الساعات المتاحة ومنع التضارب
CLINIC_WORKING_HOURS = "من السبت إلى الخميس، من الساعة 1 ظهراً حتى 10 مساءً (مدة الكشف/الجلسة ساعة واحدة: 1م، 2م، 3م، 4م، 5م، 6م، 7م، 8م، 9م). الجمعة إجازة."

st.set_page_config(page_title=f"نظام إدارة وحجز - {CLINIC_NAME}", page_icon=CLINIC_ICON, layout="wide")

st.markdown("""
<style>
    .stButton>button { border-radius: 20px; border: 1px solid #00a884; font-size: 14px; }
    .whatsapp-btn {
        display: inline-block; background-color: #25D366; color: white !important;
        padding: 10px 20px; border-radius: 25px; text-decoration: none;
        font-weight: bold; text-align: center; width: 100%; margin-top: 10px;
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
except Exception as e:
    st.error(f"برجاء التأكد من إضافة جميع المفاتيح في إعدادات Secrets. التفاصيل: {e}")
    st.stop()

# ==========================================
# 2. دوال الاتصال بـ Google Sheets (قراءة / إضافة / حذف)
# ==========================================
@st.cache_resource
def get_gsheet():
    creds = Credentials.from_authorized_user_info(
        google_creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    worksheet = sh.sheet1
    # إضافة العناوين تلقائياً لو الشيت فاضي
    if not worksheet.row_values(1):
        worksheet.append_row(["وقت التسجيل", "اسم العميل", "رقم الموبايل", "الخدمة", "اليوم", "الساعة"])
    return worksheet

def fetch_all_bookings():
    try:
        ws = get_gsheet()
        records = ws.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"تعذر قراءة Google Sheet حالياً: {e}")
        return pd.DataFrame(columns=["وقت التسجيل", "اسم العميل", "رقم الموبايل", "الخدمة", "اليوم", "الساعة"])

def add_booking_to_sheet(name, phone, service, day, time_slot):
    ws = get_gsheet()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    ws.append_row([now_str, name, str(phone), service, day, time_slot])

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

with st.spinner("جاري تحميل النظام وربط قاعدة البيانات..."):
    vector_store = load_knowledge_base()

# ==========================================
# 4. تقسيم الواجهة إلى Tab للعملاء و Tab للإدارة
# ==========================================
tab_chat, tab_admin = st.tabs(["💬 مساعد العملاء والحجز الذكي", "📊 لوحة تحكم الإدارة والمواعيد"])

# ------------------------------------------
# TAB 1: واجهة محادثة العملاء + منع تضارب المواعيد
# ------------------------------------------
with tab_chat:
    col_header, col_wa = st.columns([3, 1])
    with col_header:
        st.title(f"{CLINIC_ICON} {CLINIC_NAME}")
        st.caption("اسألني عن الأسعار، الخدمات، أو احجز ميعادك فوراً مع فحص المواعيد المتاحة.")
    with col_wa:
        wa_link = f"https://wa.me/{WHATSAPP_NUMBER}?text=أهلاً،%20بتواصل%20معاكم%20من%20المساعد%20الذكي"
        st.markdown(f'<a href="{wa_link}" target="_blank" class="whatsapp-btn">💬 واتساب مباشر</a>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"أهلاً بحضرتك في {CLINIC_NAME}! 👋 إزاي أقدر أساعدك النهاردة؟ تقدر تسأل عن الأسعار أو تطلب حجز ميعاد مباشرة."}
        ]

    # أزرار سريعة
    q_cols = st.columns(4)
    quick_question = None
    if q_cols[0].button("💰 قائمة الأسعار"): quick_question = "إيه قائمة الأسعار عندكم؟"
    if q_cols[1].button("🕒 مواعيد العمل"): quick_question = "إيه مواعيد العمل والعنوان؟"
    if q_cols[2].button("💳 أنظمة التقسيط"): quick_question = "هل متاح تقسيط أو تأمين؟"
    if q_cols[3].button("📅 حجز ميعاد"): quick_question = "عايز أحجز ميعاد"

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input("اكتب سؤالك أو تفاصيل الميعاد المطلوب هنا...")
    user_question = quick_question if quick_question else typed_question

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # جلب المواعيد المحجوزة حالياً من Google Sheets لمنع التضارب
        df_current = fetch_all_bookings()
        if not df_current.empty and "اليوم" in df_current.columns and "الساعة" in df_current.columns:
            booked_list = [f"- يوم {row['اليوم']} الساعة {row['الساعة']}" for _, row in df_current.iterrows()]
            booked_slots_text = "\n".join(booked_list)
        else:
            booked_slots_text = "لا توجد مواعيد محجوزة حالياً، جميع الساعات متاحة."

        docs = vector_store.similarity_search(user_question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        recent_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])

        prompt = f"""
        أنت موظف استقبال ومنسق مواعيد ذكي جداً في {CLINIC_NAME}.
        تحدث بالعامية المصرية المهذبة والودودة.

        === مواعيد عمل العيادة ===
        {CLINIC_WORKING_HOURS}

        === المواعيد المحجوزة بالفعل (ممنوع الحجز فيها نهائياً) ===
        {booked_slots_text}

        === بيانات العيادة والأسعار ===
        {context}

        === سجل المحادثة ===
        {recent_history}

        === رسالة العميل الحالية ===
        {user_question}

        === قواعد العمل الصارمة ===
        1. **منع تضارب المواعيد واقتراح البدائل:** إذا طلب العميل يوماً معيناً (مثل الخميس)، افحص قائمة "المواعيد المحجوزة بالفعل". إذا كانت الساعة التي يطلبها محجوزة (أو إذا سأل عن المواعيد المتاحة في ذلك اليوم)، اعتذر بلطف عن الساعات المحجوزة واحسب الساعات الفارغة المتبقية من مواعيد العمل وخير العميل بينها بوضوح.
        2. **جمع بيانات الحجز:** لتأكيد أي حجز، يجب أن تحصل من العميل على: (الاسم بالكامل، رقم الموبايل، الخدمة المطلوبة، اليوم، والساعة). لا تؤكد الحجز إلا بعد التأكد من أن اليوم والساعة غير موجودين في قائمة المواعيد المحجوزة.
        3. **تسجيل الحجز برمجياً:** بمجرد اكتمال البيانات وتأكيدك للحجز مع العميل، أضف في نهاية ردك تماماً السطر التالي بصيغة JSON دقيقة (لن يظهر للعميل):
        BOOKING_DATA: {{"name": "الاسم", "phone": "الموبايل", "service": "الخدمة", "day": "اليوم", "time": "الساعة"}}
        """

        with st.chat_message("assistant"):
            with st.spinner("جاري فحص المواعيد والرد..."):
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
                    booking_match = re.search(r"BOOKING_DATA:\s*(\{.*?\})", reply_text)
                    if booking_match:
                        try:
                            info = json.loads(booking_match.group(1))
                            add_booking_to_sheet(
                                info.get("name", "غير محدد"),
                                info.get("phone", "غير محدد"),
                                info.get("service", "كشف"),
                                info.get("day", "غير محدد"),
                                info.get("time", "غير محدد")
                            )
                            st.toast("✅ تم تسجيل الميعاد في Google Sheets بنجاح!", icon="📅")
                        except Exception as sheet_err:
                            st.error(f"تم الرد ولكن تعذر الحفظ في الشيت: {sheet_err}")
                        
                        reply_text = re.sub(r"BOOKING_DATA:\s*\{.*?\}", "", reply_text).strip()

                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                else:
                    st.error(f"تعذر الرد من الموديل. التفاصيل: {last_err}")

# ------------------------------------------
# TAB 2: لوحة تحكم الإدارة والمواعيد (CRM)
# ------------------------------------------
with tab_admin:
    st.header("📊 لوحة تحكم الإدارة وسجل المواعيد (متصل بـ Google Sheets)")
    admin_pass = st.text_input("أدخل كلمة مرور الإدارة:", type="password", placeholder="جرب: 1234")
    
    if admin_pass == ADMIN_PASSWORD:
        if st.button("🔄 تحديث البيانات من Google Sheets"):
            st.rerun()

        df_bookings = fetch_all_bookings()
        
        # إحصائيات سريعة
        m1, m2, m3 = st.columns(3)
        m1.metric("إجمالي الحجوزات المسجلة", len(df_bookings))
        m2.metric("حالة الربط السحابي", "متصل لايف 🟢")
        m3.metric("اسم الشيت", SHEET_NAME)
        
        st.divider()
        
        if not df_bookings.empty:
            st.subheader("📋 جدول المواعيد المحجوزة")
            st.dataframe(df_bookings, use_container_width=True)
            
            col_dl, col_del = st.columns([1, 2])
            with col_dl:
                csv = df_bookings.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 تحميل سجل المواعيد (Excel / CSV)",
                    data=csv,
                    file_name=f"bookings_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            with col_del:
                row_to_del = st.selectbox(
                    "إلغاء / حذف حجز (اختر رقم الصف):",
                    options=df_bookings.index,
                    format_func=lambda x: f"صف {x}: {df_bookings.loc[x, 'اسم العميل']} - {df_bookings.loc[x, 'اليوم']} ({df_bookings.loc[x, 'الساعة']})"
                )
                if st.button("🗑️ حذف الميعاد المحدد من الشيت"):
                    delete_booking_row(row_to_del)
                    st.success("تم حذف الميعاد، وأصبح متاحاً للحجز مرة أخرى!")
                    st.rerun()
        else:
            st.info("لا توجد حجوزات مسجلة في الشيت حتى الآن. جرب عمل حجز من تبويب المحادثة!")
    elif admin_pass:
        st.error("كلمة المرور غير صحيحة!")
