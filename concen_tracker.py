import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Initialisation MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Points des yeux pour le calcul EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Points pour la détection d'expressions
MOUTH_POINTS = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [296, 334, 293, 300, 276]
FOREHEAD_POINTS = [10, 151, 9, 8]

# Historiques pour le lissage
score_history = deque(maxlen=10)
mood_history = deque(maxlen=15)
distraction = 0

def eye_aspect_ratio(landmarks, eye_points, image_w, image_h):
    """Calcule le ratio d'aspect des yeux (EAR)"""
    p = []
    for idx in eye_points:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
        p.append((x, y))

    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, image_w, image_h):
    """Calcule le ratio d'aspect de la bouche pour détecter les sourires"""
    mouth_points = []
    for idx in MOUTH_POINTS:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        mouth_points.append((x, y))
    
    # Points des coins de la bouche
    left_corner = mouth_points[0]  # Point 61
    right_corner = mouth_points[1]  # Point 84
    
    # Points du haut et bas de la bouche
    top_lip = mouth_points[2]  # Point 17
    bottom_lip = mouth_points[3]  # Point 14
    
    # Calcul de la largeur et hauteur de la bouche
    mouth_width = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
    mouth_height = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))
    
    if mouth_height == 0:
        return 0
    
    mar = mouth_width / mouth_height
    return mar

def eyebrow_position(landmarks, eyebrow_points, image_w, image_h):
    """Analyse la position des sourcils pour détecter les froncements"""
    eyebrow_y = []
    for idx in eyebrow_points:
        lm = landmarks[idx]
        y = lm.y * image_h
        eyebrow_y.append(y)
    
    return np.mean(eyebrow_y)

def detect_mood(landmarks, image_w, image_h, left_ear, right_ear):
    """Détecte l'humeur basée sur les expressions faciales"""
    avg_ear = (left_ear + right_ear) / 2
    
    # Analyse de la bouche
    mar = mouth_aspect_ratio(landmarks, image_w, image_h)
    
    # Position des sourcils
    left_eyebrow_y = eyebrow_position(landmarks, LEFT_EYEBROW, image_w, image_h)
    right_eyebrow_y = eyebrow_position(landmarks, RIGHT_EYEBROW, image_w, image_h)
    avg_eyebrow_y = (left_eyebrow_y + right_eyebrow_y) / 2
    
    # Détection des expressions
    mood = "Neutre"
    confidence = 0.5
    
    # Sourire détecté
    if mar > 3.2:
        mood = "Content 😊"
        confidence = min(1.0, (mar - 3.0) / 1.0)
    
    # Yeux fermés (pensif/endormi)
    elif avg_ear < 0.15:
        mood = "Pensif 🤔"
        confidence = 1.0 - avg_ear / 0.15
    
    # Sourcils froncés (concentré/confus)
    elif avg_eyebrow_y < image_h * 0.35:
        mood = "Concentré 🧐"
        confidence = 0.8
    
    # Yeux très ouverts (surpris)
    elif avg_ear > 0.35:
        mood = "Surpris 😮"
        confidence = min(1.0, (avg_ear - 0.3) / 0.1)
    
    # Bouche légèrement ouverte (fatigué)
    elif mar > 2.8 and mar < 3.2:
        mood = "Fatigué 😴"
        confidence = 0.6
    
    return mood, confidence

def is_blinking(ear, threshold=0.2):
    """Détecte si la personne cligne des yeux"""
    return ear < threshold

def get_head_pose_score(landmarks, image_w, image_h):
    """Calcule le score de pose de la tête"""
    nose = landmarks[1]
    x = nose.x * image_w
    y = nose.y * image_h
    d = np.linalg.norm(np.array([x - image_w / 2, y - image_h / 2]))
    if d < 0.3 * image_w:  
        return 1.0
    return 0.0

def get_gaze_score(landmarks, image_w, image_h):
    """Calcule le score du regard"""
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    avg_x = (left_iris.x + right_iris.x) / 2.0
    if 0.5 < avg_x < 0.7:
        return 1.0  
    return 0.0     

def compute_concentration_score(gaze, head_pose, blink, mood_factor=1.0):
    """Calcule le score de concentration avec facteur d'humeur"""
    base_score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (0 if blink else 1)
    # L'humeur affecte la concentration
    adjusted_score = base_score * mood_factor
    return round(adjusted_score * 100, 2)

def get_mood_factor(mood):
    """Retourne un facteur multiplicateur basé sur l'humeur"""
    mood_factors = {
        "Content 😊": 1.1,      # Bonne humeur améliore la concentration
        "Concentré 🧐": 1.2,    # Déjà concentré
        "Pensif 🤔": 0.9,       # Légèrement distrait
        "Surpris 😮": 0.7,      # Distrait
        "Fatigué 😴": 0.6,      # Très distrait
        "Neutre": 1.0           # Normal
    }
    return mood_factors.get(mood.split()[0], 1.0)

def draw_concentration_bar(score, frame):
    """Dessine la barre de concentration améliorée"""
    bar_width = 200
    bar_height = 30
    bar_x = 30
    bar_y = 100
    
    # Fond de la barre
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (50, 50, 50), -1)
    
    # Remplissage selon le score
    fill_width = int(score * bar_width / 100)
    if score > 70:
        color = (0, 255, 0)  # Vert
    elif score > 40:
        color = (0, 255, 255)  # Jaune
    else:
        color = (0, 100, 255)  # Rouge
        
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + fill_width, bar_y + bar_height), 
                 color, -1)
 
    # Bordure
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (200, 200, 200), 2)
    
    # Texte du pourcentage
    cv2.putText(frame, f"{score}%", 
               (bar_x + bar_width + 10, bar_y + bar_height//2 + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

def draw_mood_info(frame, mood, confidence, image_w):
    """Affiche les informations d'humeur"""
    # Zone d'affichage de l'humeur
    mood_bg_color = (40, 40, 60)
    cv2.rectangle(frame, (image_w - 250, 100), (image_w - 30, 200), mood_bg_color, -1)
    cv2.rectangle(frame, (image_w - 250, 100), (image_w - 30, 200), (100, 100, 100), 2)
    
    # Titre
    cv2.putText(frame, "HUMEUR", (image_w - 240, 125),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Humeur détectée
    cv2.putText(frame, mood, (image_w - 240, 155),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Barre de confiance
    conf_bar_width = 150
    conf_fill = int(confidence * conf_bar_width)
    cv2.rectangle(frame, (image_w - 240, 170), (image_w - 90, 185), (50, 50, 50), -1)
    cv2.rectangle(frame, (image_w - 240, 170), (image_w - 240 + conf_fill, 185), (0, 200, 100), -1)
    cv2.putText(frame, f"Conf: {int(confidence*100)}%", (image_w - 240, 195),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)
blink_counter = 0

print("=== Système de Tracking de Concentration avec Détection d'Humeur ===")
print("Appuyez sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la vidéo")
        break

    # Interface utilisateur semi-transparente
    ui_bg = frame.copy()
    cv2.rectangle(ui_bg, (0, 0), (frame.shape[1], 150), (30, 30, 30), -1)
    cv2.addWeighted(ui_bg, 0.6, frame, 0.4, 0, frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dessiner le maillage facial
            mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=DrawingSpec(color=(0, 150, 255), thickness=1)
            )

            landmarks = face_landmarks.landmark
            
            # Calcul EAR pour les yeux
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h)
            avg_ear = (left_ear + right_ear) / 2

            # Détection du clignement
            blink = is_blinking(avg_ear)

            # Détection de l'humeur
            detected_mood, mood_confidence = detect_mood(landmarks, image_w, image_h, left_ear, right_ear)
            mood_history.append(detected_mood)
            
            # Humeur lissée
            if len(mood_history) > 5:
                most_common_mood = max(set(mood_history), key=mood_history.count)
            else:
                most_common_mood = detected_mood

            # Calcul des scores
            gaze_score = get_gaze_score(landmarks, image_w, image_h)
            head_score = get_head_pose_score(landmarks, image_w, image_h)
            mood_factor = get_mood_factor(most_common_mood)
            concentration = compute_concentration_score(gaze_score, head_score, blink, mood_factor)

            # Lissage du score de concentration
            score_history.append(concentration)
            smooth_score = int(np.mean(score_history))
            
            # Affichage des barres et informations
            draw_concentration_bar(smooth_score, frame)
            draw_mood_info(frame, most_common_mood, mood_confidence, image_w)

            # Texte principal de concentration
            cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Indicateur de clignement
            if blink:
                cv2.putText(frame, "CLIGNEMENT", (30, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

            # Gestion de la distraction
            if smooth_score < 40:
                distraction += 1
                cv2.putText(frame, f"Distraction: {distraction}", (30, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                if distraction > 1000:
                    distraction = 0
                    print('Système désactivé - Trop de distraction')
            else:
                distraction = max(0, distraction - 2)  # Récupération graduelle
    
    # Affichage des FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (image_w - 120, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    
    # Indicateur de statut
    status_color = (0, 255, 0) if distraction < 50 else (0, 100, 255)
    cv2.circle(frame, (image_w - 30, 70), 15, status_color, -1)
    status_text = "ACTIF" if distraction < 50 else "DISTRAIT"
    cv2.putText(frame, status_text, (image_w - 120, 75), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Affichage de la fenêtre
    cv2.imshow("Trackeur de Concentration + Humeur", frame)
    
    # Gestion de la sortie
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
print("Système arrêté")