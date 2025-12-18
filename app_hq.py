import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
from ultralytics import SAM
from streamlit_drawable_canvas import st_canvas
import json
import re

# --- 1. アプリ設定 ---
st.set_page_config(page_title="AI Room Segmentation HQ", layout="wide")
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stButton>button {width: 100%; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🏠 AI Room Segmentation (高精度対応版)")
st.caption("モデルを変更して精度を向上させることができます")

# --- 2. サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # --- Geminiモデル選択 ---
    available_models = []
    selected_gemini_model = ""
    if api_key:
        try:
            genai.configure(api_key=api_key)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                # flash系を優先選択
                default_idx = 0
                for i, name in enumerate(available_models):
                    if "flash" in name and "1.5" in name:
                        default_idx = i
                        break
                selected_gemini_model = st.selectbox("1. Geminiモデル (頭脳)", available_models, index=default_idx)
            else:
                st.error("利用可能なモデルが見つかりません")
        except:
            pass

    st.markdown("---")
    
    # --- SAMモデル選択 (ここが新機能) ---
    st.markdown("### 2. 切り抜き精度 (目)")
    sam_type = st.radio(
        "精度を選択してください",
        ["高速 (MobileSAM)", "高精度 (SAM Base)"],
        captions=["速い・粗い (40MB)", "遅い・綺麗 (370MB)"],
        index=0
    )

# --- 3. モデルローダー ---
@st.cache_resource
def load_sam_model(model_type):
    if model_type == "高速 (MobileSAM)":
        return SAM('mobile_sam.pt')
    else:
        # 高精度モデル (初回はダウンロードに時間がかかります)
        return SAM('sam_b.pt')

try:
    # 選択されたモデルをロード
    sam_model = load_sam_model(sam_type)
    if sam_type == "高精度 (SAM Base)":
        st.sidebar.success("✨ 高精度モデル使用中")
    else:
        st.sidebar.info("🚀 高速モデル使用中")
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
    st.stop()


# --- 4. メイン処理 ---
def process_gemini_auto(image, api_key, model_name):
    genai.configure(api_key=api_key)
    width, height = image.size
    
    prompt = """
    この画像の「天井(Ceiling)」「壁(Wall)」「床(Floor)」を検出してください。
    窓やドアがある場合は壁とは区別して除外するか、壁に含めるか判断してください。
    出力は以下のJSON形式のみ（Markdownなし）で行ってください。
    座標は画像サイズに対する 0〜1000 の正規化座標 [ymin, xmin, ymax, xmax] です。
    
    [
        {"label": "Ceiling", "box_2d": [ymin, xmin, ymax, xmax]},
        {"label": "Wall", "box_2d": [ymin, xmin, ymax, xmax]},
        {"label": "Floor", "box_2d": [ymin, xmin, ymax, xmax]}
    ]
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, image])
        
        text_resp = response.text
        match = re.search(r'\[.*\]', text_resp, re.DOTALL)
        if not match:
            return None, f"JSONが見つかりませんでした。応答: {text_resp[:100]}..."
            
        json_data = json.loads(match.group(0))
        
        bboxes = []
        labels = []
        for item in json_data:
            ymin, xmin, ymax, xmax = item["box_2d"]
            box = [
                xmin / 1000 * width,
                ymin / 1000 * height,
                xmax / 1000 * width,
                ymax / 1000 * height
            ]
            bboxes.append(box)
            labels.append(item.get("label", "Object"))
            
        if bboxes:
            # SAM推論
            results = sam_model(image, bboxes=bboxes)
            return results[0], f"成功 (Model: {model_name})"
        else:
            return None, "検出対象なし"
        
    except Exception as e:
        return None, str(e)

def main():
    uploaded_file = st.file_uploader("部屋の写真をアップロード", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        width, height = image_pil.size
        
        tab1, tab2 = st.tabs(["🤖 全自動モード", "👆 手動指定モード"])
        
        with tab1:
            st.write("選択されたモデルで解析します。")
            if st.button("🚀 解析スタート", key="auto"):
                if not api_key or not selected_gemini_model:
                    st.error("APIキーとモデルを選択してください")
                else:
                    with st.spinner(f"解析中... (高精度モードは少し時間がかかります)"):
                        result, info = process_gemini_auto(image_pil, api_key, selected_gemini_model)
                    
                    if result:
                        col1, col2 = st.columns(2)
                        with col1: st.image(image_pil, caption="元画像", use_column_width=True)
                        with col2: st.image(result.plot(), caption="解析結果", use_column_width=True)
                        st.success(info)
                    else:
                        st.error(f"エラー: {info}")

        with tab2:
            st.write("手動モード (API不要)")
            
            canvas_width = 700
            scale = canvas_width / width if width > canvas_width else 1.0
            d_w, d_h = int(width * scale), int(height * scale)

            col_c, col_d = st.columns([2, 1])
            with col_c:
                canvas = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=image_pil,
                    update_streamlit=True,
                    height=d_h,
                    width=d_w,
                    drawing_mode="rect",
                    key="canvas_hq",
                )

            if canvas.json_data and canvas.json_data["objects"]:
                obj = canvas.json_data["objects"][-1]
                scale_x, scale_y = width / d_w, height / d_h
                
                box = [
                    obj["left"] * scale_x,
                    obj["top"] * scale_y,
                    (obj["left"] + obj["width"]) * scale_x,
                    (obj["top"] + obj["height"]) * scale_y
                ]
                
                with col_d:
                    with st.spinner("切り抜き中..."):
                        res = sam_model(image_pil, bboxes=[box])
                        st.image(res[0].plot(), caption="切り抜き結果", use_column_width=True)

if __name__ == "__main__":
    main()
