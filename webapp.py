import PIL
from cv2 import normalize
import streamlit as st
import os
import datetime
VISITOR_COUNT_DIR = "visitor_counts"



def classification(compare_img):
    import numpy as np
    import cv2
    
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

    return top_matches, orb, des2, compare_img, kp2, matches_data

def download(preresize):
    import numpy as np
    import cv2
    
    import glob
    import io
    from PIL import Image

    from reportlab.pdfgen import canvas
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    # Create a PDF canvas

    current_date = get_current_date()
    pdf = canvas.Canvas('유사도 분석 결과'+ ' ' +f'{current_date}.pdf')

    pdfmetrics.registerFont(TTFont('SUITE', r'SUITE-Light.ttf'))

    # Set the page dimensions
    page_width = 595.276  # A4 page width in points
    page_height = 841.890  # A4 page height in points

    # Calculate the image and caption positions
    image_width = 400
    image_height = 300
    image_x = (page_width - image_width) / 2
    image_y = (page_height - image_height) / 2
    # caption_y = image_y - 20
    page_square = page_width, page_height
    image_square = image_width, image_height
    image_coordinate = image_x, image_y
    fill_pdf(preresize, pdf, page_square, image_square, image_coordinate)

    


def wannasee(preresize):
    import numpy as np
    import cv2
    
    import glob
    import io
    from PIL import Image
    preresize = preresize
    result = classification(preresize)
    top_matches, orb, des2, compare_img, kp2, matches_data = result

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
        st.image(Image.open(io.BytesIO(img_bytes)), caption=f'비슷한 이미지 #{i+1} (유사도: {similarity:.2f})') # 비활성화


       
def fill_pdf(preresize, pdf, page_square, image_square, image_coordinate):
    import numpy as np
    import cv2
    
    import glob
    import io
    from PIL import Image
    result = classification(preresize)
    top_matches, orb, des2, compare_img, kp2, matches_data = result
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
        page_width, page_height = page_square
        image_width, image_height = image_square
        image_x, image_y = image_coordinate
        caption_y = image_y - 20
        # Convert OpenCV image to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

        # Save PIL image as a temporary file
        temp_filename = f'temp_image_{i}.png'
        pil_image.save(temp_filename)

        # Add the image to the PDF canvas
        pdf.drawInlineImage(temp_filename, image_x, image_y, width=image_width, height=image_height)

        # Add the caption below the image
        caption = f'비슷한 이미지 #{i+1} (유사도: {similarity:.2f})'
        pdf.setFont('SUITE', 12)  # Use the registered custom font
        caption_width = pdf.stringWidth(caption, 'SUITE', 12)
        caption_x = (page_width - caption_width) / 2
        pdf.drawString(caption_x, caption_y, caption)

        # Delete the temporary file
        os.remove(temp_filename)

        # Add a new page if there are more images
        if i < len(matches_data) - 1:
            pdf.showPage()
        

    
    st.code("다운로드가 시작됩니다.", language= 'java')
    # Save the PDF file
    pdf = pdf.save()
    st.download_button(data= pdf)






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
    
    
    st.metric(label="방문자 수", value=user, delta=user_check(date))

    





def usage():
    import streamlit as st
    

    st.write("사용법 소개")

def based_information():
    import streamlit as st
    
    
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    
    st.header("이 앱에서 판별 가능한 암의 종류들")
    st.write("- ['편평 세포암'](https://en.wikipedia.org/wiki/Squamous-cell_carcinoma) ")
    st.write("- ['기저 세포암'](https://en.wikipedia.org/wiki/Basal-cell_carcinoma) ")
    st.write("- ['양성 각화증'](https://en.wikipedia.org/wiki/Seborrheic_keratosis) ")
    st.write("- ['피부 섬유종'](https://en.wikipedia.org/wiki/Dermatofibroma)  ")
    st.write("- ['멜라닌 세포 모반'](https://en.wikipedia.org/wiki/Melanocytic_nevus)  ")
    st.write("- ['흑색종'](https://en.wikipedia.org/wiki/Melanoma) ")
    st.write("- ['혈관 병변'](https://en.wikipedia.org/wiki/Vascular_anomaly) ")
    st.write(" ")




def testing():
    import streamlit as st
    
    import time
    import tensorflow as tf
    
    import numpy as np
    




    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path="models/mobilenet_v1_1.0_224_quant.tflite")
    interpreter.allocate_tensors()
    interpreter = tf.lite.Interpreter(model_path="models/InceptionResNetV2Skripsi.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # 이미지 업스케일링 


   # default upsc

    # Define a function to resize the input image
    # def resize_image(image):
    #     # Resize the image to 150x150 pixels
    #     resized_image = tf.image.resize(image, [150, 150])
        
    #     return resized_image.numpy()


    # replace upsc


    def resize_image(image):
        from PIL import Image
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

    # Define the labels for the 7 classes
    labels = ['편평 세포암', '기저 세포암', '양성 각화증', '피부 섬유종', '흑색종', '멜라닌 세포 모반', '혈관 병변증']

    from PIL import Image
    from urllib.error import URLError

    

    st.warning("이미지 제출 방식을 골라주세요")

    upload = st.checkbox("파일 업로드하기")
    take = st.checkbox("사진 찍기")
    if upload:
        image = st.file_uploader(label="파일 업로드하기", type=["jpg", "jpeg", "png"])
        if image is not None:
            # image_upscailing(image)
            image = np.array(Image.open(image).convert("RGB"))
            preresize = image
            st.code("작업에 사용된 이미지", language= "java")
            st.image(image, width=150)

            # Run inference on the input image
            probs, classifying_duration = classify_image(image)

            
            # Display the classification duration
            st.success(f"소요 시간: {classifying_duration:.4f} 초")

            # Display the top 3 predictions
            top_3_indices = np.argsort(probs)[::-1][:3]
            
            st.write("예측되는 상위 3개의 징후:")
            for i in range(3):
                st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
            st.divider()
            st.code("유사점 비교", language='python')
            
            classification(preresize)
            st.button("다시보기")
    elif take:
        image = st.camera_input(label="사진 찍기", help="웹캠 지원")
        if image is not None:
            
            image = np.array(Image.open(image).convert("RGB"))
            preresize = image
            

            # Run inference on the input image
            probs, classifying_duration = classify_image(image)

            
            # Display the classification duration
            st.success(f"소요 시간: {classifying_duration:.4f} 초")

            # Display the top 3 predictions
            top_3_indices = np.argsort(probs)[::-1][:3]
            
            st.write("예측되는 상위 3개의 징후:")
            for i in range(3):
                st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
            st.subheader(":blue[유사도 비교] 아래 중복선택 가능")
            

            ws = st.checkbox("유사도 비교 결과 보기 :exclamation: 피부암 사진이 직접 노출됩니다.")
            dl = st.checkbox("유사도 비교 결과 다운로드 하기 :exclamation: 피부암 사진이 직접 노출됩니다.")
            if ws:    
                wannasee(preresize)
            if dl:
                download(preresize)
            st.button("다시보기")

            # 다운로드

        

# 페이지 구성 

page_names_to_funcs = {
    "홈": intro,
    "피부암에 대한 기본 정보": based_information,
    "사용법": usage,
    "검사하기": testing
}

page_name = st.sidebar.selectbox("원하는 페이지를 고르세요", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()