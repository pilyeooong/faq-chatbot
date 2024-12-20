from llm import get_ai_response
import streamlit as st

TITLE = "스마트스토어 FAQ 챗봇"
CAPTION = "스마트스토어 이용에 궁금한 점을 모두 물어보세요 !"

st.set_page_config(page_title=TITLE, page_icon="😀")

st.title(TITLE)
st.caption(CAPTION)

if 'message_list' not in st.session_state:
    st.session_state['message_list'] = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="스마트스토어 이용에 궁금한 점을 모두 물어보세요 !"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_response(st.session_state["message_list"], user_question)

        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "assistant", "content": ai_message})