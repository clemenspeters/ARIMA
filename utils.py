def anomalies_index_to_window_index(anomalies, window_size, stride = 0):
    if stride == 0:
        stride = window_size / 2
    anomaly_windows = []
    for anomaly in anomalies:
        window = int(anomaly / stride)
        if (window - 1) not in anomaly_windows:
            anomaly_windows.append(window - 1)
        if window not in anomaly_windows:
            anomaly_windows.append(window)
    # return set(anomaly_windows)
    return anomaly_windows