import cv2
import mediapipe as mp

# inisiliasi mediapipe tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
# fungsi hitung jari yg d angkat
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # indeks tinggi ujung jari
    fingers_up = 0
    # ibu jari
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers_up += 1
    # 4 jari yg lain
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up += 1
    
    return fingers_up
# buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    # konversi ke RGB untuk mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect tangan
    results = hands.process(image_rgb)
    # jika detect tangan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # gambar landmark tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)            
            # hitung jumlah jari yg d angkat
            num_fingers = count_fingers(hand_landmarks)
            # nampilin jumlah jari
            cv2.putText(image, f'Jari: {num_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # buat biar ga minnor kameranya
    image = cv2.flip(image, 1)
    # nampilin hasil
    cv2.imshow('Traking Tangan', image)
    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# bersihkan
cap.release()
cv2.destroyAllWindows()