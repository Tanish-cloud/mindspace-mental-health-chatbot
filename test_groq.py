from groq import Groq

import os
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def get_groq_response(user_text, detected_emotion, confidence_scores, is_audio=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add chat history
    for msg in st.session_state.chat_history:
        messages.append({
            "role": msg.role,
            "content": msg.parts[0].text
        })
    
    # Add current message
    messages.append({"role": "user", "content": f"""
Detected emotion: {detected_emotion}
User said: "{user_text}"
Confidence — Depression: {confidence_scores.get('Depression',0):.1%}, 
Anxiety: {confidence_scores.get('Anxiety',0):.1%}, 
Stress: {confidence_scores.get('Stress',0):.1%}
Please respond as Mindify."""})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=400,
        temperature=0.75
    )
    return response.choices[0].message.content