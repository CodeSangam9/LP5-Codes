import cv2
import os
import time

DATASET_DIR = "face_dataset"
IMG_SIZE = (100, 100)
SAMPLES = 20

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face(gray):
    return face_cascade.detectMultiScale(gray, 1.3, 5)


def collect_data(person_name):
    path = os.path.join(DATASET_DIR, person_name)
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    print(f"\nCollecting {SAMPLES} images for {person_name}...")

    count = 0
    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_face(gray)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMG_SIZE)

            file_path = os.path.join(path, f"{count}.jpg")
            cv2.imwrite(file_path, face)

            print(f"Saved {file_path}")
            count += 1

            time.sleep(0.5)  # auto delay

        cv2.imshow(f"Collecting - {person_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def check_dataset():
    if not os.path.exists(DATASET_DIR):
        return False

    persons = os.listdir(DATASET_DIR)
    if len(persons) < 2:
        return False

    for p in persons:
        if len(os.listdir(os.path.join(DATASET_DIR, p))) < 10:
            return False

    return True


def main():
    print("\n===== FACE DATA COLLECTION SYSTEM =====")

    if not check_dataset():
        print("\nDataset not ready → Auto collecting now")

        collect_data("Person1")
        collect_data("Person2")

    print("\n✅ Dataset Ready!")

    # Show dataset info
    for p in os.listdir(DATASET_DIR):
        count = len(os.listdir(os.path.join(DATASET_DIR, p)))
        print(f"{p} → {count} images")


if __name__ == "__main__":
    main()