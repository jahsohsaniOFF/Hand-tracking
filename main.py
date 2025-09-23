# main.py — Détection main + comptage doigts (pouce robuste)
import cv2, time, mediapipe as mp, math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Indices MediaPipe
TIP = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
PIV = [3, 6, 10, 14, 18]   # Thumb IP pour le pouce, PIP pour autres doigts
MCP = [2, 5, 9, 13, 17]    # MCP joints
WRIST = 0

def dist(a, b):
    dx, dy = a.x - b.x, a.y - b.y
    return math.hypot(dx, dy)

def fingers_up(hand_lm, handedness_label):
    """
    Règles :
      - Index/Majeur/Annulaire/Auriculaire : tip.y < PIV.y  (plus haut = levé)
      - Pouce (robuste) : 3 tests combinés
          A) le tip est plus éloigné du poignet que l’articulation IP (extension)
          B) écart latéral du tip par rapport au MCP du pouce significatif
          C) le tip est du même côté latéral que le MCP du pouce relativement au MCP de l’index
    """
    lm = hand_lm.landmark

    # Doigts autres que le pouce
    index_up  = lm[TIP[1]].y < lm[PIV[1]].y
    middle_up = lm[TIP[2]].y < lm[PIV[2]].y
    ring_up   = lm[TIP[3]].y < lm[PIV[3]].y
    pinky_up  = lm[TIP[4]].y < lm[PIV[4]].y

    # --- Pouce robuste ---
    tip  = lm[TIP[0]]     # 4
    ip   = lm[PIV[0]]     # 3
    mcp  = lm[MCP[0]]     # 2
    wrist= lm[WRIST]      # 0
    idxm = lm[MCP[1]]     # MCP index (5)

    # A) Extension (tip plus loin du poignet que IP)
    ext_ok = dist(tip, wrist) > dist(ip, wrist) * 1.05

    # B) Écart latéral du tip par rapport au MCP du pouce (seuil ~ 0.03 en coords normalisées)
    lateral_spread = abs(tip.x - mcp.x) > 0.03

    # C) Cohérence du côté par rapport à l’index (évite faux positifs quand pouce replié)
    #    On regarde si tip et mcp sont du même côté de l’axe passant par MCP du pouce et MCP de l’index.
    side_tip = tip.x - idxm.x
    side_mcp = mcp.x - idxm.x
    same_side = (side_tip * side_mcp) > 0

    thumb_up = ext_ok and lateral_spread and same_side

    details = {
        "thumb": thumb_up,
        "index": index_up,
        "middle": middle_up,
        "ring": ring_up,
        "pinky": pinky_up
    }
    return sum(details.values()), details

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Webcam introuvable.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    prev_t = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)  # miroir pour confort
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                labels = [c.classification[0].label for c in (res.multi_handedness or [])]
                for i, hlm in enumerate(res.multi_hand_landmarks):
                    mp_draw.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    lab = labels[i] if i < len(labels) else "Hand"
                    count, d = fingers_up(hlm, lab)

                    text = f"{lab}: {count} doigt(s)"
                    if d["thumb"] and not any([d["index"], d["middle"], d["ring"], d["pinky"]]):
                        text += " | Pouce levé 👍"

                    cv2.putText(frame, text, (10, 30 + i*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1]-130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("Mains + comptage doigts (q pour quitter)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
