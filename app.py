
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def health():
    return jsonify({"ok": True, "service": "artemo-openai-backend"})

@app.get("/diag")
def diag():
    has_key = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY").startswith("sk-")
    return jsonify({"has_openai_key": has_key})


@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "image required"}), 400

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一名艺术治疗师，负责分析画作中体现的情绪（非医疗诊断）。请仅返回JSON，包含summary、emotions、salient_features、interventions、risk、meta。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请按指定schema输出："},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ]
        )
        result = resp.choices[0].message.content.strip()
        return jsonify({"analysis": result})
    except Exception as e:
        app.logger.exception("openai failed")
        return jsonify({"error": "AI request failed"}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
