import os
import openai
import groq
from dotenv import load_dotenv

load_dotenv()

def generate_rationale(rec_obj):
    """
    Generates a plain-English rationale for the recommended action.
    Uses a 3-tier fallback chain: OpenAI -> Groq -> Template.
    """
    current_risk_score = f"{rec_obj.get('current_risk_score', 0) * 100:.1f}%"
    recommended_action = rec_obj.get('recommended_action', 'Unknown action')
    predicted_new_risk = f"{rec_obj.get('predicted_new_risk', 0) * 100:.1f}%"
    
    evidence = rec_obj.get('supporting_evidence', {})
    historical_precedent = evidence.get('historical_precedent', 'No historical precedent available.')
    causal_pathway = evidence.get('causal_pathway', 'No causal pathway specified.')
    leading_indicator_context = rec_obj.get('leading_indicator_context', 'Unknown instability.')
    
    prompt = f"""You are helping a paper mill operator understand an automated recommendation. Given:
- Current risk of off-spec basis weight: {current_risk_score}
- Recommended action: {recommended_action}
- Predicted risk after action: {predicted_new_risk}
- Historical precedent: {historical_precedent}
- Causal basis: {causal_pathway}
- Current instability signal: {leading_indicator_context}

Write ONE plain-English sentence (max 30 words) explaining what's happening and why this action is recommended. Do NOT claim to know the root cause of the disturbance — only describe rising instability and the grounded reason the action should help. Do not invent information not provided."""

    # TIER 1: OpenAI
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not set")
            
        client = openai.OpenAI(timeout=5.0)
        
        # Dynamically fetch available models
        model_list_response = client.models.list(timeout=5.0)
        available_models = [m.id for m in model_list_response.data]
        
        # Pick cheapest chat model available
        if 'gpt-4o-mini' in available_models:
            chosen_model = 'gpt-4o-mini'
        elif 'gpt-3.5-turbo' in available_models:
            chosen_model = 'gpt-3.5-turbo'
        else:
            chosen_model = 'gpt-4o'
            
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=5.0
        )
        return {
            "rationale": response.choices[0].message.content.strip(),
            "source_tier": "openai"
        }
    except Exception as e:
        print(f"[Rationale Layer] OpenAI fallback triggered due to: {e}")
        pass # Fall through to Tier 2
        
    # TIER 2: Groq
    try:
        if 'GROQ_API_KEY' not in os.environ:
            raise ValueError("GROQ_API_KEY not set")
            
        gclient = groq.Groq(timeout=5.0)
        response = gclient.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{"role": "user", "content": prompt}],
            timeout=5.0
        )
        return {
            "rationale": response.choices[0].message.content.strip(),
            "source_tier": "groq"
        }
    except Exception as e:
        print(f"[Rationale Layer] Groq fallback triggered due to: {e}")
        pass # Fall through to Tier 3
        
    # TIER 3: Template
    try:
        # Build a safe template string
        template = f"With risk at {current_risk_score} due to {leading_indicator_context.lower()}, we recommend to {recommended_action.lower()} because this historically stabilized similar cases and relies on the {causal_pathway.split(' ')[0]} causal link."
        return {
            "rationale": template,
            "source_tier": "template"
        }
    except Exception as e:
        return {
            "rationale": "High risk detected; apply the recommended action based on historical precedent.",
            "source_tier": "template_fallback"
        }

if __name__ == "__main__":
    # Test cases
    rec_16 = {
      "current_risk_score": 0.5520,
      "recommended_action": "Reduce stock_flow by 0.1%",
      "predicted_new_risk": 0.1120,
      "supporting_evidence": {
        "historical_precedent": "Similar to 5 historical cases, most of which stabilized by adjusting stock_flow.",
        "causal_pathway": "stock_flow has a validated causal effect on basis_weight."
      },
      "source": "historical_success",
      "leading_indicator_context": "Steam pressure variability indicates rising instability."
    }
    
    rec_34 = {
      "current_risk_score": 0.5515,
      "recommended_action": "Increase steam_pressure by 0.1%",
      "predicted_new_risk": 0.0763,
      "supporting_evidence": {
        "historical_precedent": "Similar to 5 historical cases, most of which stabilized by adjusting steam_pressure.",
        "causal_pathway": "steam_pressure has a validated causal effect on moisture."
      },
      "source": "historical_success",
      "leading_indicator_context": "Steam pressure variability indicates rising instability."
    }
    
    print("--- Event 16 ---")
    res = generate_rationale(rec_16)
    print(f"Tier: {res['source_tier']}")
    print(f"Rationale: {res['rationale']}")
    
    print("\n--- Event 34 ---")
    res = generate_rationale(rec_34)
    print(f"Tier: {res['source_tier']}")
    print(f"Rationale: {res['rationale']}")
    
    print("\n--- Fallback Testing (Removing Keys) ---")
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    res = generate_rationale(rec_16)
    print(f"Tier (No OpenAI): {res['source_tier']}")
    
    if 'GROQ_API_KEY' in os.environ:
        del os.environ['GROQ_API_KEY']
    res = generate_rationale(rec_16)
    print(f"Tier (No Groq): {res['source_tier']}")
