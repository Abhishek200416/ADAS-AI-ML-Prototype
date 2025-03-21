def collision_avoidance(fused_data, distance_threshold=10):
    """
    Decide on actions based on the fused sensor data.
    If the simulated radar distance is below the threshold, output "Brake".
    """
    actions = []
    radar_distance = fused_data['radar']['distance']
    for det in fused_data['detections']:
        if radar_distance < distance_threshold:
            action = 'Brake'
        else:
            action = 'Continue'
        actions.append({'class': det['class'], 'action': action})
    return actions

if __name__ == '__main__':
    # Simulated fused data example
    fused_data = {
        'frame_number': 10,
        'detections': [
            {'bbox': [72.04, 208.04, 338.84, 333.11], 'confidence': 0.90, 'class': 'car'},
            {'bbox': [684.18, 190.07, 793.35, 270.82], 'confidence': 0.88, 'class': 'car'}
        ],
        'radar': {'distance': 13.17, 'relative_speed': 0.81}
    }
    decisions = collision_avoidance(fused_data)
    print("Decision Actions:")
    for d in decisions:
        print(d)
