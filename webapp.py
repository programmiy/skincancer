import time
import streamlit as st
import os
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image

VISITOR_COUNT_DIR = "visitor_counts"





       
        
        




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

    from markdownlit import mdlit

    st.success("제작에 도움을 준 사이트들")

    mdlit(
        """




@(https://streamlit.io)

[orange]Inspired by[/orange] @(https://huggingface.co/ilhamstoked)

""")
    

    
    mdlit(
        """[blue]Our[/blue] [green]Repository[/green]      @(https://github.com/programmiy/skincancer)
        """
    )
    

    from streamlit_extras.badges import badge

    badge(type="github", name="programmiy/skincancer")
    def get_visitor_count(date):
        file_path = os.path.join(VISITOR_COUNT_DIR, f"{date}.txt")

        if not os.path.exists(file_path):
            return 0

        with open(file_path, "r") as f:
            count = int(f.read())
            return count
    
    date = get_current_date()   
    user = get_visitor_count(date)
    
    
    def get_delta(now): #파라미터 없애기, 혹은  get_current_date 수정
        now = datetime.datetime.now()
        delta = now - datetime.timedelta(days= 1)
        delta = delta.strftime("%Y-%m-%d") # get_current_date랑 겹침
        
        return delta
        
    def user_check(date):
        delta = get_delta(date)
        if get_visitor_count(date) - get_visitor_count(delta) < 0:
            return 0
        else:
            return get_visitor_count(date) - get_visitor_count(delta)
    
    
    st.metric(label="방문자 수", value=user, delta=user_check(date), help="하루 동안 이 사이트에 방문한 방문자 수입니다.")

    from streamlit_extras.metric_cards import style_metric_cards
    style_metric_cards()
    



def usage():

    

    st.write("사용법 소개")

def based_information():
    
    from markdownlit import mdlit


    
    
    
  

    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Wikipedia_wordmark.svg/600px-Wikipedia_wordmark.svg.png')
    st.header("이 앱에서 판별 가능한 암의 종류들")
    # st.write(f"- 편평 세포암 {Squamous_cell}")
    mdlit(
        """- @([orange]편평 세포암[/orange])(https://en.wikipedia.org/wiki/Squamous-cell_carcinoma) ->  사마귀나 물집 등의 형태나 검은색 덩어리가 융기하여 나타나는 피부암
        """
    )
    mdlit(
        """- @([orange]기저 세포암[/orange])(https://en.wikipedia.org/wiki/Basal-cell_carcinoma) ->  피부 안에서 부터 발전하는 피부암, 기저층이나 모낭 등을 구성하는 세포가 악성화한 종양
        """
    )

    mdlit(
        """- @([orange]양성 각화증[/orange])(https://en.wikipedia.org/wiki/Seborrheic_keratosis) ->  검버섯이라고 부르는, 표피 각질층에 구성된 사마귀 모양의 흔한 피부양성종양이다. """
    )
    
    mdlit(
        """- @([orange]피부 섬유종[/orange])(https://en.wikipedia.org/wiki/Dermatofibroma) ->  피부가 상처가 난 뒤에 자주 발생하는 양성 종양 """
    )
    mdlit(
        """- @([orange]멜라닌 세포 모반[/orange])(https://en.wikipedia.org/wiki/Melanocytic_nevus) ->  흑색종과 가장 비슷한 양성 종양 """
    )
    mdlit(
        """- @([orange]흑색종[/orange])(https://en.wikipedia.org/wiki/Melanoma) -> 멜라닌 세포 모반이 비대칭이거나 경계가 뚜렷하지 않은 악성 종양  """
    )
    mdlit(
        """- @([orange]혈관 병변[/orange])(https://en.wikipedia.org/wiki/Vascular_anomaly)  -> 모세혈관의 확장으로 생기는 병변, 피부암은 아니며  보라색, 자주색, 빨간색 등  색깔은 다양한 편"""
    )




    # 이미지 업스케일링 


   # default upsc

    # Define a function to resize the input image
    # def resize_image(image):
    #     # Resize the image to 150x150 pixels
    #     resized_image = tf.image.resize(image, [150, 150])
        
    #     return resized_image.numpy()


    # replace upsc


def resize_image(image):
    
    """
    Pads the input image and resizes it to 150x150 pixels.

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The padded and resized image tensor.

    Raises:
        ValueError: If the input image has a shape smaller than 150x150 pixels.
    """
    # Get the original image shape
    original_shape = tf.shape(image)
    
    # Calculate the padding amounts
    height_pad = (original_shape[0] - 150) // 2
    width_pad = (original_shape[1] - 150) // 2
    
    # Pad the image
    padded_image = tf.pad(image, [[height_pad, height_pad], [width_pad, width_pad], [0, 0]])
    
    # Resize the padded image to 150x150 pixels
    resized_image = tf.image.resize(padded_image, [150, 150])

    
    return resized_image

    # Define a function to run inference on the TensorFlow Lite model
def classify_image(image):
        
    """
        Runs inference on the input image using the provided interpreter and model details.

        Args:
            image (numpy.ndarray): The input image array.
            interpreter (tensorflow.lite.Interpreter): The TensorFlow Lite interpreter.
            input_details (list): A list of input details dictionaries for the interpreter.
            output_details (list): A list of output details dictionaries for the interpreter.

        Returns:
            tuple: A tuple containing the output probabilities and the classification duration.

        Raises:
            ValueError: If the input image has an invalid format or shape.
        
    """            
        # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path="models/mobilenet_v1_1.0_224_quant.tflite")
    interpreter.allocate_tensors()
    interpreter = tf.lite.Interpreter(model_path="models/InceptionResNetV2Skripsi.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Pre-process the input image
    resized_image = resize_image(image)
    

    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    
    
    # Run inference
    with st.spinner('분석 중...'):
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
    
    # Calculate the classification duration
    classifying_duration = end_time - start_time
   
    # Get the output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])
    

    return output_data[0], classifying_duration
def testing():


    st.warning("이미지 제출 방식을 골라주세요")

    upload = st.checkbox("파일 업로드하기")
    take = st.checkbox("사진 찍기")
    if upload:
        image = st.file_uploader(label="파일 업로드하기", type=["jpg", "jpeg", "png"])
        if image is not None:
            compute_analysis(image)
            # # image_upscailing(image)
            # image = np.array(Image.open(image).convert("RGB"))
            # preresize = image
            # st.code("작업에 사용된 이미지", language= "java")
            # st.image(image, width=150)

            # # Run inference on the input image
            # probs, classifying_duration = classify_image(image)

            
            # # Display the classification duration
            # st.success(f"소요 시간: {classifying_duration:.4f} 초", icon="✅")
            
            # # Display the top 3 predictions
            # top_3_indices = np.argsort(probs)[::-1][:3]
            
            # st.write("예측되는 상위 3개의 징후:")
            # for i in range(3):
            #     st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
            # st.divider()
            # st.code("유사점 비교", language='python')
            
            # classification(preresize)
                

    elif take:
        image = st.camera_input(label="사진 찍기", help="웹캠 지원")
        if image is not None:
            compute_analysis(image)

def collapse(img_bytes_array, similarity_list): # nested collapse
    
    
    i = 0
    for img_bytes, similarity in zip(img_bytes_array, similarity_list):
        st.image(Image.open(img_bytes), caption=f'비슷한 이미지 #{i+1} (유사도: {similarity:.2f})')
        i +=1 
            # 다운로드

def compute_analysis(image): # 중복작업 일괄처리용

    """
    Computes analysis for the input image, including classification and visual comparison.

    Args:
        image (str): The path to the input image.

    Returns:
        None
    """    
    import numpy as np
    
    from PIL import Image
    # Define the labels for the 7 classes
    labels = ['편평 세포암', '기저 세포암', '양성 각화증', '피부 섬유종', '흑색종', '멜라닌 세포 모반', '혈관 병변증']
        
    image = np.array(Image.open(image).convert("RGB"))
    preresize = image
            

    # Run inference on the input image
    probs, classifying_duration = classify_image(image) # + upsc

            
    # Display the classification duration
    st.success(f"소요 시간: {classifying_duration:.4f} 초", icon="✅")

    # Display the top 3 predictions
    top_3_indices = np.argsort(probs)[::-1][:3]
            
    st.write("예측되는 상위 3개의 징후:")
    for i in range(3):
        st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
    with st.expander("유사점 비교해보기 "):
            
        st.header("유사점 비교")
        imgarray, similarity_list = classification(preresize)
                
                # Convert HTML to image using imgkit
                

        wannasee = st.button("결과 직접 보기")
        if wannasee:
                    
            collapse(imgarray, similarity_list)
                
    st.button("다시보기")

def classification(compare_img):
    import numpy as np
    import cv2
    
    import glob
    import io
    from PIL import Image

    import base64
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

    imgarray = []
    similarity_list = []
    # Generate HTML content for download
    html_content = "<html><body>"
    for i, (img_path, similarity) in enumerate(top_matches):
        base_img = cv2.imread(img_path)
        kp1, des1 = orb.detectAndCompute(base_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(base_img, kp1, compare_img, kp2, matches[:10], None, flags=2)
        img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        
        

        # Save the image as BytesIO object
        img_bytes = io.BytesIO()
        Image.fromarray(img3_rgb).save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        imgarray.append(img_bytes)
        similarity_list.append(similarity)

        # Add the image to the HTML content
        html_content += f"<h3>비슷한 이미지 #{i+1} (유사도: {similarity:.2f})</h3>"
        html_content += f"<img src='data:image/png;base64,{base64.b64encode(img_data).decode()}'/><br><br>"

    html_content += "</body></html>"

    # Download button
    st.download_button("결과를 html로 다운로드 받기 ", data=html_content, file_name="분석 결과.html", mime="text/html")

    return  imgarray, similarity_list



    # # Display the matched images
    # for i, (img_path, similarity) in enumerate(top_matches):
    #     base_img = cv2.imread(img_path)
    #     kp1, des1 = orb.detectAndCompute(base_img, None)

    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     matches = bf.match(des1, des2)
    #     matches = sorted(matches, key=lambda x: x.distance)

    #     img3 = cv2.drawMatches(base_img, kp1, compare_img, kp2, matches[:10], None, flags=2)
        
    #     # Create a Streamlit image object from the OpenCV image
    #     # st.write(f"Base image shape: {base_img.shape}")
    #     # st.write(f"Compare image shape: {compare_img.shape}")
    #     img_bytes = cv2.imencode('.png', img3)[1].tobytes()
    #     st.image(Image.open(io.BytesIO(img_bytes)), caption=f'비슷한 이미지 #{i+1} (유사도: {similarity:.2f})') # 비활성화

# 페이지 구성 

page_names_to_funcs = {
    "홈": intro,
    "피부암에 대한 기본 정보": based_information,
    "사용법": usage,
    "검사하기": testing
}

page_name = st.sidebar.selectbox("원하는 페이지를 고르세요", page_names_to_funcs.keys())



page_names_to_funcs[page_name]()
