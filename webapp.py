import streamlit as st
import os
import datetime
VISITOR_COUNT_DIR = "visitor_counts"



def classification(compare_img):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    import glob
    import io
    from PIL import Image

    img_list = glob.glob('cancer/*.jpg')
    compare_img = np.array(compare_img)
    compare_img = cv2.cvtColor(compare_img, cv2.COLOR_RGB2BGR) 
    

    orb = cv2.ORB_create()
    kp2, des2 = orb.detectAndCompute(compare_img, None)

    matches_data = []
    for img_path in img_list:
        base_img = cv2.imread(img_path, 0)
        
        kp1, des1 = orb.detectAndCompute(base_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if matches:
            similarity = sum(match.distance for match in matches) / len(matches)
        else:
            similarity = 0

        matches_data.append((img_path, similarity))

    # sort matches by similarity score
    matches_data.sort(key=lambda x: x[1], reverse=True)
    
    # get top 5 matches
    top_matches = matches_data[:5]

    # Display the matched images
    for i, (img_path, similarity) in enumerate(top_matches):
        base_img = cv2.imread(img_path)
        kp1, des1 = orb.detectAndCompute(base_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(base_img, kp1, compare_img, kp2, matches[:10], None, flags=2)
        
        # Create a Streamlit image object from the OpenCV image
        # st.write(f"Base image shape: {base_img.shape}")
        # st.write(f"Compare image shape: {compare_img.shape}")
        img_bytes = cv2.imencode('.png', img3)[1].tobytes()
        st.image(Image.open(io.BytesIO(img_bytes)), caption=f'비슷한 이미지 #{i+1} (유사도: {similarity:.2f})')

    


def get_current_date():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d")






def intro():
    import streamlit as st
    def increase_visitor_count():
        current_date = get_current_date()
        file_path = os.path.join(VISITOR_COUNT_DIR, f"{current_date}.txt")

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("0")

        with open(file_path, "r+") as f:
            count = int(f.read())
            count += 1
            f.seek(0)
            f.write(str(count))
    increase_visitor_count()
    st.write("# 피부암 검사를 위한 페이지에 온것을 환영합니다👨‍⚕️")
    st.markdown("""
    **👈 사이드바를 클릭해서** 더 많은 정보를 확인하세요!
    """)
    st.sidebar.success("원하는 정보를 확인하세요")
    code = '''powered by Streamlit'''
    st.code(code, language='java')
    st.subheader("⚠️ 의학적 지식을 대체하는 것이 아닙니다 ")
    st.subheader("⚠️ 잠재적인 피부 문제를 쉽게 파악하는 도움을 받을 수 있도록 돕는 도구라는 점에 유의해야 합니다")
    st.markdown(
        """

        

        ### 페이지 제작 도움 크레딧

        - Check out [streamlit.io](https://streamlit.io)
        - big thanks 
        - 새로운거


        """)
    def get_visitor_count(date):
        file_path = os.path.join(VISITOR_COUNT_DIR, f"{date}.txt")

        if not os.path.exists(file_path):
            return 0

        with open(file_path, "r") as f:
            count = int(f.read())
            return count
    
    date = get_current_date()   
    user = get_visitor_count(date)
    now = datetime.datetime.now()
    
    def get_delta(now):
        now = datetime.datetime.now()
        delta = now - datetime.timedelta(days= 1)
        delta = delta.strftime("%Y-%m-%d")
        
        return delta
        
    def user_check(date):
        delta = get_delta(date)
        return get_visitor_count(date) - get_visitor_count(delta)
    
    st.info("# 오류 제보하기 ")
    def on_submit():
        title = st.session_state.title
        details = st.session_state.details
        st.write(f"title: {title}, details: {details}")

    with st.form(key="my_form"):
        st.text_input("제목", key="title")
        st.text_area("내용", key="details")
        st.form_submit_button("제출하기", on_click=on_submit, type= "primary")
    st.metric(label="방문자 수", value=user, delta=user_check(date))







def usage():
    import streamlit as st
    
    st.write ("6.	The application also measures the latency or response time for the model to classify the image, and displays it to the user.")
    st.write("Please note that this model still has room for academic revision as it can only classify the following 7 classes")
    st.write("- ['akiec'](https://en.wikipedia.org/wiki/Squamous-cell_carcinoma) - squamous cell carcinoma (actinic keratoses dan intraepithelial carcinoma),")
    st.write("- ['bcc'](https://en.wikipedia.org/wiki/Basal-cell_carcinoma) - basal cell carcinoma,")
    st.write("- ['bkl'](https://en.wikipedia.org/wiki/Seborrheic_keratosis) - benign keratosis (serborrheic keratosis),")
    st.write("- ['df'](https://en.wikipedia.org/wiki/Dermatofibroma) - dermatofibroma, ")
    st.write("- ['nv'](https://en.wikipedia.org/wiki/Melanocytic_nevus) - melanocytic nevus, ")
    st.write("- ['mel'](https://en.wikipedia.org/wiki/Melanoma) - melanoma,")
    st.write("- ['vasc'](https://en.wikipedia.org/wiki/Vascular_anomaly) - vascular skin (Cherry Angiomas, Angiokeratomas, Pyogenic Granulomas.)")
    st.write("Due to imperfection of the model and a room of improvement for the future, if the probabilities shown are less than 70%, the skin is either healthy or the input image is unclear. This means that the model can be the first diagnostic of your skin illness. As precautions for your skin illness, it is better to do consultation with dermatologist. ")

def based_information():
    import streamlit as st
    import time
    import numpy as np
    
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)
    
   
        

    

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("다시보기")


def testing():
    import streamlit as st
    import pandas as pd
    import altair as alt
    import time
    import tensorflow as tf
    from tqdm import tqdm
    import numpy as np
    import pandas as pd 


    # Measure Latency
    #18.155.181.8
    #35.201.127.49:443
    #192.168.18.6:8501

    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224_quant.tflite")
    interpreter.allocate_tensors()
    interpreter = tf.lite.Interpreter(model_path="InceptionResNetV2Skripsi.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define a function to resize the input image
    def resize_image(image):
        # Resize the image to 150x150 pixels
        resized_image = tf.image.resize(image, [150, 150])
        
        return resized_image.numpy()

    # Define a function to run inference on the TensorFlow Lite model
    def classify_image(image):
        # Pre-process the input image
        resized_image = resize_image(image)
        input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        
        
        # Run inference
        with st.spinner('Classifying...'):
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            
        
        # Calculate the classification duration
        classifying_duration = end_time - start_time
       
        # Get the output probabilities
        output_data = interpreter.get_tensor(output_details[0]['index'])
        

        return output_data[0], classifying_duration

    # Define the labels for the 7 classes
    labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    from PIL import Image
    from urllib.error import URLError

    

    st.warning("이미지 제출 방식을 골라주세요")

    upload = st.checkbox("파일 업로드하기")
    take = st.checkbox("사진 찍기")
    if upload:
        image = st.file_uploader(label="파일 업로드하기", type=["jpg", "jpeg", "png"])
        if image is not None:
            image = np.array(Image.open(image).convert("RGB"))
            preresize = image
            st.image(image, width=150)

            # Run inference on the input image
            probs, classifying_duration = classify_image(image)

            
            # Display the classification duration
            st.success(f"소요 시간: {classifying_duration:.4f} 초")

            # Display the top 3 predictions
            top_3_indices = np.argsort(probs)[::-1][:3]
            
            st.write("피부암으로 추정되는 3개의 증상:")
            for i in range(3):
                st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
            st.divider()
            st.code("유사점 비교", language='python')
            
            classification(preresize)
    elif take:
        image = st.camera_input(label="사진 찍기", help="웹캠 지원")
        if image is not None:
            image = np.array(Image.open(image).convert("RGB"))
            preresize = image
            st.image(image, width=150)

            # Run inference on the input image
            probs, classifying_duration = classify_image(image)

            
            # Display the classification duration
            st.success(f"소요 시간: {classifying_duration:.4f} 초")

            # Display the top 3 predictions
            top_3_indices = np.argsort(probs)[::-1][:3]
            
            st.write("피부암으로 추정되는 3개의 증상:")
            for i in range(3):
                st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
            st.subheader("#유사점 비교")
            classification(preresize)


        

    

page_names_to_funcs = {
    "홈": intro,
    "피부암에 대한 기본 정보": based_information,
    "사용법": usage,
    "검사하기": testing
}

demo_name = st.sidebar.selectbox("원하는 페이지를 고르세요", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()