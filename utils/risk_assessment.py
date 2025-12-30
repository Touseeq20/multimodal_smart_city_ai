
def calculate_risk(incident_type, confidence, details):
    """
    Calculates the risk level based on the incident type and details.
    
    Args:
        incident_type (str): Type of incident (e.g., 'Traffic Accident', 'Fire', 'Congestion')
        confidence (float): Confidence score of the detection (0-1).
        details (dict): Metadata about the incident (e.g., vehicle_count).
        
    Returns:
        str: 'LOW', 'MEDIUM', 'HIGH'
    """
    incident_type = incident_type.lower()
    
    if any(word in incident_type for word in ['fire', 'accident', 'overturned', 'wreckage', 'debris', 'gridlock']):
        return 'HIGH'
            
    if 'congestion' in incident_type:
        vehicle_count = details.get('vehicle_count', 0)
        if vehicle_count > 10:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    # Default fallback
    return 'LOW'
