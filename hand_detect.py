import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk menghitung jari yang diangkat
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Indeks titik ujung jari
    fingers_up = 0
    
    # Thumb (Ibu jari)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers_up += 1
    
    # Other four fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up += 1
    
    return fingers_up

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Konversi ke RGB untuk MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Deteksi tangan
    results = hands.process(image_rgb)
    
    # Jika mendeteksi tangan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar landmark tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hitung jumlah jari yang diangkat
            num_fingers = count_fingers(hand_landmarks)
            
            # Gambarkan garis di layar berdasarkan jumlah jari yang diangkat
            start_point = (50, 100)  # Titik awal untuk menggambar garis
            end_point = (50 + num_fingers * 50, 100)  # Titik akhir berdasarkan jumlah jari
            color = (255, 0, 0)  # Warna garis (Biru)
            thickness = 5  # Ketebalan garis
            
            # Gambar garis
            cv2.line(image, start_point, end_point, color, thickness)
            
            # Tampilkan jumlah jari yang diangkat
            cv2.putText(image, f'Fingers: {num_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # buat biar ga minnor kameranya
    image = cv2.flip(image, 1)

    # Tampilkan hasil
    cv2.imshow('Hand Tracking', image)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()