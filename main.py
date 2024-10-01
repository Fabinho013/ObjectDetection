import cv2
import numpy as np

# Configurações
thres = 0.45  # Threshold de confiança
nms_threshold = 0.2  # Threshold para NMS (Non-Maximum Suppression)

# Iniciar captura de vídeo (0 para webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Largura do frame
cap.set(4, 480)  # Altura do frame
cap.set(10, 150)  # Brilho da imagem

# Carregar nomes das classes
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# Carregar os arquivos do modelo
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Carregar o modelo de detecção
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Loop para processar o vídeo
while True:
    success, img = cap.read()
    if not success:
        break

    # Detectar objetos
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Converter para listas
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Aplicar Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    
    # Desenhar caixas e adicionar rótulos
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Desenhar a caixa delimitadora
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

            # Adicionar rótulo e confiança
            cv2.putText(img, f'{classNames[classIds[i][0] - 1].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar o frame com detecções
    cv2.imshow("Object Detection", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
