Suivi de Concentration

Un système de suivi de concentration en temps réel développé avec MediaPipe et OpenCV. Cet outil évalue l'attention d’un utilisateur à partir des clignements des yeux, de la direction du regard et de l’orientation de la tête — idéal pour des applications comme le suivi d’étude, l’e-learning ou l’amélioration de la productivité.
Fonctionnalités

    Détection des clignements d’yeux
    Calcule le ratio d’aspect des yeux (EAR) pour détecter les clignements et les périodes de fermeture prolongée des paupières.

    Détection du regard
    Estime si l’utilisateur regarde droit devant ou ailleurs à l’aide des points de repère de l’iris.

    Estimation de la pose de la tête
    Évalue l’orientation de l’utilisateur en fonction de la position du nez par rapport au centre de l’écran.

    Score de concentration
    Calcule un score pondéré combinant la direction du regard, la pose de la tête et le comportement de clignement des yeux.

    Retour visuel en temps réel
    Interface superposée à la vidéo de la webcam affichant le niveau de concentration, l’état de clignement et un compteur de distractions.

    Suivi des distractions
    Compte le nombre d’images où l’utilisateur ne prête pas attention et affiche des alertes si nécessaire.

Exemple d'affichage

Le flux vidéo affiche :

    Une barre de pourcentage de concentration

    Des alertes de détection de clignement

    Un compteur de distractions

    Un indicateur ACTIF / DISTRAIT

    Un compteur FPS

Pile technologique

    Python 3.x

    OpenCV

    MediaPipe (FaceMesh)

    NumPy

Fonctionnement

    Les points du visage sont détectés grâce à MediaPipe FaceMesh.

    Le EAR (Eye Aspect Ratio) est utilisé pour détecter les clignements.

    La position de l’iris permet d’évaluer la direction du regard.

    La position du nez sert à inférer la pose de la tête.

    Un score de concentration composite est calculé selon la formule :
    score = 0.4 * regard + 0.4 * pose_tête + 0.2 * (pas de clignement)

    Un système de retour visuel affiche en temps réel le niveau de concentration de l’utilisateur.
## Run the Project

```bash
git clone https://github.com/asutoshp10/concentration_tracker.git
cd concentration_tracker
pip install -r requirements.txt
python concentration_tracker.py
