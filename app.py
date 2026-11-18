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

# ==========================================
# 1. إعدادات الهوية والبيزنس (تتغير بسهولة لكل عميل)
# ==========================================
CLINIC_NAME = "عيادة زايد سمايل للأسنان"
CLINIC_ICON = "🦷"
WHATSAPP_NUMBER = "201001234567"  # رقم الواتساب بصيغة دولية بدون +
PDF_FILE_NAME = "clinic_data.pdf"
ADMIN_PASSWORD = "1234"  # كلمة سر لوحة تحكم صاحب البيزنس

st.set_page_config(page_title=f"المساعد الذكي - {CLINIC_NAME}", page_icon=CLINIC_ICON, layout="centered")

# تحسين شكل الأزرار والواجهة
st.markdown("""
<style>
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #00a884;
        font-size: 14px;
    }
    .whatsapp-btn {
        display: inline-block;
        background-color: #25D366;
        color: white !important;
        padding: 10px 20px;
        border-radius: 25px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        width: 100%;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# قراءة مفتاح الـ API
try:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    st.error("عذراً، لم يتم ضبط مفتاح التشغيل (GEMINI_API_KEY) في إعدادات الاستضافة.")
    st.stop()

# ==========================================
# 2. تجهيز قاعدة المعرفة (RAG)
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
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

with st.spinner("جاري تجهيز المساعد الذكي..."):
    vector_store = load_knowledge_base()

# تهيئة سجل الحجوزات (Lead Capture Storage)
if "bookings" not in st.session_state:
    st.session_state.bookings = [
        {"الوقت": "2026-11-18 10:30", "اسم العميل": "أحمد محمود (تجريبي)", "رقم الموبايل": "01012345678", "الخدمة": "تبييض أسنان", "الميعاد المفضل": "الخميس 6 مساءً"}
    ]

# ==========================================
# 3. الشريط الجانبي: لوحة تحكم صاحب البيزنس + واتساب
# ==========================================
with st.sidebar:
    st.header(f"{CLINIC_ICON} {CLINIC_NAME}")
    st.write("مساعد ذكي متكامل للرد على الاستفسارات وحجز المواعيد 24/7.")
    
    wa_msg = "أهلاً، بتواصل معاكم من خلال المساعد الذكي على الموقع."
    wa_link = f"https://wa.me/{WHATSAPP_NUMBER}?text={wa_msg.replace(' ', '%20')}"
    st.markdown(f'<a href="{wa_link}" target="_blank" class="whatsapp-btn">💬 تواصل مباشر عبر واتساب</a>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("🔒 لوحة تحكم الإدارة (للمالك)")
    admin_pass = st.text_input("أدخل كلمة السر لعرض الحجوزات:", type="password", placeholder="جرب: 1234")
    
    if admin_pass == ADMIN_PASSWORD:
        st.success(f"إجمالي الحجوزات المسجلة: {len(st.session_state.bookings)}")
        df_bookings = pd.DataFrame(st.session_state.bookings)
        st.dataframe(df_bookings, use_container_width=True)
        
        csv = df_bookings.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 تحميل الحجوزات (Excel / CSV)",
            data=csv,
            file_name="clinic_bookings.csv",
            mime="text/csv",
        )
    elif admin_pass:
        st.error("كلمة السر غير صحيحة")

# ==========================================
# 4. واجهة المحادثة الرئيسية + الأزرار السريعة
# ==========================================
st.title(f"{CLINIC_ICON} المساعد الذكي - {CLINIC_NAME}")
st.caption("اسألني عن الأسعار، المواعيد، أنظمة التقسيط، أو احجز ميعادك مباشرة هنا!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"أهلاً بحضرتك في {CLINIC_NAME}! 👋 إزاي أقدر أساعدك النهاردة؟ تقدر تسألني عن أي خدمة أو تحجز ميعادك مباشرة."}
    ]

# أزرار الأسئلة السريعة (Quick Replies)
cols = st.columns(4)
quick_question = None
if cols[0].button("💰 قائمة الأسعار"):
    quick_question = "إيه قائمة الأسعار عندكم؟"
if cols[1].button("🕒 مواعيد العمل"):
    quick_question = "إيه مواعيد العمل والعنوان؟"
if cols[2].button("💳 أنظمة التقسيط"):
    quick_question = "هل متاح تقسيط أو تأمين طبي؟"
if cols[3].button("📅 حجز ميعاد"):
    quick_question = "عايز أحجز ميعاد"

# عرض سجل المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال السؤال سواء من الكتابة أو من الأزرار السريعة
typed_question = st.chat_input("اكتب سؤالك أو بيانات الحجز هنا...")
user_question = quick_question if quick_question else typed_question

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # البحث في قاعدة المعرفة
    docs = vector_store.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # تجهيز ذاكرة آخر 6 رسائل لفهم السياق
    recent_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])

    # تعليمات الذكاء الاصطناعي للرد + استخراج بيانات الحجز أوتوماتيك
    prompt = f"""
    أنت موظف استقبال ومبيعات محترف وودود في {CLINIC_NAME}.
    أجب على العميل بالعامية المصرية المهذبة اعتماداً على السياق وسجل المحادثة.
    
    قواعد العمل:
    1. لا تؤلف أي أسعار أو خدمات غير موجودة في السياق.
    2. إذا أراد العميل الحجز، اطلب منه بلطف: (الاسم بالكامل، رقم الموبايل، الخدمة المطلوبة، واليوم/الميعاد المناسب).
    3. إذا أعطاك العميل اسمه ورقم موبايله لتأكيد الحجز، رحب به وأكد له تسجيل بياناته، وفي نهاية ردك تماماً أضف السطر البرمجي التالي بدقة لاستخراج البيانات (لن يظهر للعميل):
    BOOKING_DATA: {{"name": "اسم العميل", "phone": "رقم الموبايل", "service": "الخدمة", "time": "الميعاد"}}

    السياق (بيانات العيادة):
    {context}

    سجل المحادثة السابقة (للحفاظ على السياق):
    {recent_history}

    رسالة العميل الحالية:
    {user_question}
    """

    with st.chat_message("assistant"):
        with st.spinner("بيكتب..."):
            client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
            reply_text = None
            
            try:
                available_models = [
                    m.name.replace("models/", "")
                    for m in client.models.list()
                    if "generateContent" in (m.supported_actions or [])
                    and "gemini" in m.name.lower()
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
                    except Exception:
                        continue
            except Exception as e:
                st.error(f"حدث خطأ في الاتصال: {e}")

            if reply_text:
                # فحص هل البوت استخرج حجز جديد أم لا
                booking_match = re.search(r"BOOKING_DATA:\s*(\{.*?\})", reply_text)
                if booking_match:
                    try:
                        booking_info = json.loads(booking_match.group(1))
                        st.session_state.bookings.append({
                            "الوقت": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "اسم العميل": booking_info.get("name", "غير محدد"),
                            "رقم الموبايل": booking_info.get("phone", "غير محدد"),
                            "الخدمة": booking_info.get("service", "كشف عام"),
                            "الميعاد المفضل": booking_info.get("time", "غير محدد")
                        })
                        st.toast("✅ تم تسجيل الحجز في لوحة تحكم الإدارة!", icon="🎉")
                    except Exception:
                        pass
                    # إخفاء الكود البرمجي عن عين العميل
                    reply_text = re.sub(r"BOOKING_DATA:\s*\{.*?\}", "", reply_text).strip()

                st.markdown(reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})
