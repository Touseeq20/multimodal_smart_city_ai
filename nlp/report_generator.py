
from transformers import pipeline

class ReportGenerator:
    def __init__(self):
        """
        Initializes the text generation pipeline.
        Upgraded to 'google/flan-t5-base' for professional research-level reasoning.
        """
        print("Loading NLP model (Flan-T5-Base)...")
        # task='text2text-generation'
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def generate_report(self, incident_type, details, risk_level):
        """
        Generates a natural language report based on structured data.
        """
        # We formulate a prompt for the model
        # Flan-T5 is good at following instructions.
        
        vehicle_count = details.get('vehicle_count', 0)
        
        # Creating a structured input prompt
        # Extract Flags (Specific visual cues)
        flags = details.get("flags", [])
        flags_context = f" VISUAL CUES: {', '.join(flags)}." if flags else ""

        # Creating an advanced instruction-following prompt for professional output
        if "Normal" in incident_type:
            prompt = f"TASK: Act as a Smart City Traffic Monitor. Write a professional report. CONTEXT: {vehicle_count} vehicles detected, traffic flow is stable."
        elif "Congestion" in incident_type:
            prompt = f"TASK: Urban Safety Alert. Write an urgent congestion report. CONTEXT: {vehicle_count} vehicles detected. Congestion is high. RISK: {risk_level}."
        elif "Accident" in incident_type:
            prompt = f"TASK: Emergency Management System. Write a CRITICAL incident report. CONTEXT: A severe incident ({incident_type}) has occurred.{flags_context} {vehicle_count} vehicles and {details.get('person_count', 0)} persons present. RISK: {risk_level}."
        elif "Fire" in incident_type:
            prompt = f"TASK: Safety Warning. Write a critical alert for public safety. CONTEXT: Fire/Smoke detected on camera. RISK: {risk_level}."
        else:
            prompt = f"TASK: Smart City Report. Status: {incident_type}. Context: {details}."

        # Generate text with repetition penalties and diverse sampling for professional flow
        outputs = self.generator(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2
        )
        return outputs[0]['generated_text']

if __name__ == "__main__":
    # Test
    rg = ReportGenerator()
    print(rg.generate_report("Traffic Congestion", {"vehicle_count": 25}, "HIGH"))
