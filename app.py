import streamlit as st
import torch
import time
import numpy as np
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import base64

# 使用 Streamlit 的缓存机制来缓存模型和处理器
@st.cache_resource
def load_model_and_processor():
    model_dir = "/home/qwen_vl_chat_demo/model/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor

model, processor = load_model_and_processor()

# 检查模型所在设备
device = next(model.parameters()).device
st.write(f"Model is loaded on device: {device}")

# 使用 Streamlit 的缓存机制来缓存图片的 base64 编码
@st.cache_data(ttl=3600)
def encode_image_base64(image_data):
    return base64.b64encode(image_data).decode('utf-8')

def extract_frames(video_path, num_frames=1):
    """从视频中抽取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frames.append(frame)
    cap.release()
    return frames

def generate_response(video_frames, user_text):
    # 构建消息结构
    video_paths = [f"data:image/jpeg;base64,{encode_image_base64(cv2.imencode('.jpg', frame)[1].tobytes())}" for frame in video_frames]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_paths,
                    "fps": 1.0,
                },
                {"type": "text", "text": user_text}
            ]
        }
    ]
    
    # 准备推理数据
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 确保有图像或视频输入
    if not image_inputs and not video_inputs:
        raise ValueError("image, image_url or video should in content.")
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    # 确保所有输入张量都在与模型相同的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 记录开始时间
    start_time = time.time()

    # 推理并生成输出
    with torch.no_grad():  # 避免梯度计算带来的数值不稳定
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_k=50)
            
            # 检查生成的概率分布
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            if torch.any(probs.sum(dim=-1) <= 0):
                raise ValueError("Invalid probability distribution (sum of probabilities <= 0)")
            
        except RuntimeError as e:
            return "", -1
    
    # 记录结束时间
    end_time = time.time()
    
    # 处理生成结果中的数值不稳定问题
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # 计算总耗时
    elapsed_time = end_time - start_time
    
    return output_text[0], elapsed_time

# Streamlit 应用界面
st.title('多模态对话Demo')

# 导航栏上传图片或视频
uploaded_file = st.sidebar.file_uploader("上传一张图片或一段视频", type=["jpg", "jpeg", "png", "mp4", "avi"])
user_input = st.sidebar.text_input("请输入描述请求")

# 左侧栏：图片或视频预览
left_col, right_col = st.columns(2)

with left_col:
    if uploaded_file is not None:
        # 显示上传的内容
        if uploaded_file.type.startswith('image'):
            # 显示上传的图片
            st.image(uploaded_file, caption='上传的图片', use_column_width=True)
            
            # 缓存图片的 base64 编码
            image_data = uploaded_file.read()
            image_base64 = encode_image_base64(image_data)
            image_url = f"data:image/jpeg;base64,{image_base64}"
            video_frames = [cv2.imdecode(np.frombuffer(image_data, np.uint8), 1)]
        elif uploaded_file.type.startswith('video'):
            # 显示上传的视频
            st.video(uploaded_file)
            
            # 提取视频帧
            video_path = uploaded_file.name
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.read())
            video_frames = extract_frames(video_path, num_frames=1)
        
        if st.sidebar.button('生成描述'):
            if video_frames and user_input:
                # 生成回复
                response, elapsed_time = generate_response(video_frames, user_input)
                
                # 在右侧栏展示结果
                with right_col:
                    if response:
                        st.write('**分析结果:**', response)
                        st.write(f'**耗时:** {elapsed_time:.2f} 秒')

# 右侧栏：展示图片或视频分析结果
with right_col:
    if uploaded_file is not None and user_input:
        # 生成回复
        response, elapsed_time = generate_response(video_frames, user_input)
        if response:
            st.write('**分析结果:**', response)
            st.write(f'**耗时:** {elapsed_time:.2f} 秒')