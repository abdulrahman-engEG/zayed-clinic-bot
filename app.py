import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types

# إعداد واجهة الصفحة بشكل احترافي للعميل
st.set_page_config(page_title="مساعد عيادة زايد سمايل الذكي", page_icon="🦷")
st.title("🦷 المساعد الذكي - عيادة زايد سمايل")
st.caption("أهلاً بك! اسألني عن الأسعار، مواعيد العمل، طرق التقسيط، أو خدمات العيادة.")

# قراءة مفتاح الـ API من الأسرار المحفوظة في السيرفر
try:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    st.error("عذراً، لم يتم ضبط مفتاح التشغيل (GEMINI_API_KEY) في إعدادات الاستضافة.")
    st.stop()

# 1. قراءة وتجهيز ملف الـ PDF المحفوظ في المشروع تلقائياً
@st.cache_resource
def load_knowledge_base():
    pdf_reader = PdfReader("clinic_data.pdf")
    raw_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

with st.spinner("جاري تجهيز المساعد الذكي..."):
    vector_store = load_knowledge_base()

# 2. واجهة المحادثة (Chat UI)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "أهلاً بحضرتك في عيادة زايد سمايل! إزاي أقدر أساعدك النهاردة؟"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("اكتب سؤالك هنا (مثال: بكام تبييض الأسنان؟)..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    docs = vector_store.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    أنت موظف استقبال وخدمة عملاء محترف وودود في عيادة زايد سمايل للأسنان.
    أجب على سؤال العميل بالعامية المصرية المهذبة اعتماداً فقط على المعلومات الموجودة في السياق التالي.
    إذا لم تجد الإجابة في السياق، قل بلطف: "عذراً، المعلومة دي مش متوفرة عندي حالياً، تقدر تتواصل معانا على رقم 01001234567 للتفاصيل." ولا تؤلف أي معلومة.

    السياق (بيانات العيادة):
    {context}

    سؤال العميل:
    {user_question}
    """

    with st.chat_message("assistant"):
        with st.spinner("بيكتب..."):
            client = genai.Client(
                api_key=api_key,
                http_options={'api_version': 'v1beta'}
            )
            
            reply_text = None
            last_error = ""
            
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
                    except Exception as e:
                        last_error = f"({model_name}): {str(e)}"
                        continue
            except Exception as list_err:
                last_error = f"ListModels Error: {str(list_err)}"
            
            if reply_text:
                st.markdown(reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})
            else:
                st.error(f"تعذر الاتصال. التفاصيل: {last_error}")
