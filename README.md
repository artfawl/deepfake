# deepfake

запуск:
python main.py -[flags]
Описание флагов:
-r путь до root файла (папка с .py файлами, и папками с картинками и масками)
-im путь до папки с картинками (исключая root составляющую)
-ma путь до папки с масками (все маски в одной папке, при распаковки .zip файла маски расфасованы по разным папкам, их надо объединить)
-a train/test начать тренировку или протестировать (для тестрирования должен быть файл model.pth)
-d cpu/cuda девайс для обучения
-e кол-во эпох для тренировки
-bs размер батча
Пример запуска:
!python3 /content/main.py -r /content -im /CelebAMask-HQ/CelebA-HQ-img -ma /CelebAMask-HQ/CelebAMask-HQ-mask-anno/all_masks -a train -d cuda
(запускал из colab)

Что использовал:
Модель deeplabv3 с backbone resnet50
Loss-функция - focal-loss

Результаты:
dice score of class skin is 0.7799643278121948
dice score of class hair is 0.259548157453537
dice score of class neck is 0.4082545340061188
dice score of class neck_l is 1.894798755645752
dice score of class r_eye is 0.0467836894094944
dice score of class r_brow is 0.07533875852823257
dice score of class hat is 1.9153372049331665
dice score of class ear_r is 1.4914973974227905
dice score of class l_lip is 0.009219302795827389
dice score of class u_lip is 0.009971318766474724
dice score of class r_ear is 1.0227460861206055
dice score of class mouth is 0.8529614210128784
dice score of class l_ear is 0.9221154451370239
dice score of class cloth is 0.7625946998596191
dice score of class eye_g is 1.8786028623580933
dice score of class l_brow is 0.0561365969479084
dice score of class l_eye is 0.045193225145339966
dice score of class nose is 0.0030028021428734064


Много экспериментов провести не удалось в виду ограниченности ресурсов
(обучал на colab бесплатной версии, где ограничивают доступ к gpu), по
этой же причине само обучение длилось не более 2х эпох.
Из опробованного:
Модельки из torchvision.models.segmentation
bce loss и focal loss
Выбрал focal loss, так как есть возможность небольшого видоизменения
для учёта дисбаланса классов на картинке (skin занимает много больше места, чем eye), 
хотя идею балансировки не удалось довести до ума, поэтому метрики по классам, которые занимают
малую часть изображения оставляют желать лучшего.

Так же не использовал scheduler'ы и не занимался подбором гиперпараметров. 

