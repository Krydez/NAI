"""
Program do detekcji gestów dłoni z wykorzystaniem OpenCV i cvzone.
cvzone wykorzystuje MediaPipe do detekcji punktów charakterystycznych dłoni.

Obsługuje 4 gesty palcami:
- 2 palce - play/pause
- 3 palce - następna piosenka
- 4 palce - poprzednia piosenka
- 5 palców - zrzut ekranu

pip:

pip install opencv-python mediapipe==0.10.9 cvzone pyautogui pillow

Autorzy: Kacper Olejnik, Hubert Jóżwiak
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import time
from datetime import datetime
import os

class GestureDetector:
    def __init__(self):
        # Inicjalizacja detektora dłoni z cvzone
        self.detector = HandDetector(detectionCon=0.7, maxHands=1)
        
        # Opóźnienie między akcjami (aby uniknąć wielokrotnego wykonania)
        self.last_action_time = 0
        self.action_cooldown = 1.5  # sekundy
        
        # Folder na zrzuty ekranu
        self.screenshots_folder = "screenshots"
        if not os.path.exists(self.screenshots_folder):
            os.makedirs(self.screenshots_folder)
    
    def perform_action(self, finger_count):
        """
        Wykonuje akcję na podstawie liczby palców
        """
        current_time = time.time()
        
        # Sprawdzenie czy minął czas cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        action = None
        
        if finger_count == 2:
            # Play/Pauza (używamy playpause - standardowy klawisz multimedialny)
            pyautogui.press('playpause')
            action = "⏯️ Play/Pauza"
            
        elif finger_count == 3:
            # Następna piosenka
            pyautogui.press('nexttrack')
            action = "⏭️ Następna piosenka"
            
        elif finger_count == 4:
            # Poprzednia piosenka
            pyautogui.press('prevtrack')
            action = "⏮️ Poprzednia piosenka"
            
        elif finger_count == 5:
            # Zrzut ekranu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.screenshots_folder, f"screenshot_{timestamp}.png")
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)
            action = f"📸 Zrzut ekranu: {screenshot_path}"
        
        if action:
            self.last_action_time = current_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {action}")
        
        return action
    
    def run(self):
        """
        Główna pętla programu
        """
        print("🔧 Inicjalizacja kamery...")
        # Inicjalizacja kamery (0 = domyślna kamera)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Użyj DirectShow na Windows
        
        # Ustawienie rozdzielczości dla Logitech C920
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Nie można otworzyć kamery!")
            print("⚠️ Sprawdź czy kamera jest podłączona i dostępna")
            return
        
        print("🎥 Kamera uruchomiona")
        print("👋 Program detekcji gestów - gotowy!")
        print("\nGesty:")
        print("  2 palce (✌️)  - Play/Pauza")
        print("  3 palce (🤟) - Następna piosenka")
        print("  4 palce (🖖) - Poprzednia piosenka")
        print("  5 palców (🖐️) - Zrzut ekranu")
        print("\nNaciśnij 'q' aby zakończyć\n")
        
        current_action = None
        
        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Nie można odczytać klatki z kamery")
                break
            
            # Odbicie lustrzane dla lepszego UX
            frame = cv2.flip(frame, 1)
            
            # Detekcja dłoni
            hands, frame = self.detector.findHands(frame)
            
            finger_count = 0
            
            # Jeśli wykryto dłoń
            if hands:
                hand = hands[0]  # Pierwsza dłoń
                fingers = self.detector.fingersUp(hand)  # Lista [kciuk, wskazujący, środkowy, serdeczny, mały]
                
                # Liczymy wyprostowane palce
                finger_count = fingers.count(1)
                
                # Wykonanie akcji (tylko dla 2-5 palców)
                if 2 <= finger_count <= 5:
                    action = self.perform_action(finger_count)
                    if action:
                        current_action = action
            
            # Wyświetlanie informacji na ekranie
            cv2.putText(frame, f"Palce: {finger_count}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if current_action:
                # Usuń emoji z wyświetlania (mogą powodować błędy w cv2.putText)
                display_action = current_action.replace("📸", "").replace("⏯️", "").replace("⏭️", "").replace("⏮️", "").strip()
                cv2.putText(frame, f"Akcja: {display_action}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Instrukcje
            cv2.putText(frame, "Nacisnij 'q' aby zakonczyc", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Wyświetlanie obrazu
            cv2.imshow('Detekcja Gestow', frame)
            
            # Wyjście z programu
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Sprzątanie
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Program zakończony")


if __name__ == "__main__":
    detector = GestureDetector()
    detector.run()
