import csv
import cv2


class allWriter(object):

    def __init__(self, config, videoName):
        self.videoName = videoName
        self.userId = config.userId
        self.epochCNN = config.epochCNN
        self.epochRL = config.epoch_rl
        self.log_path = config.log_path

        CSVfilenameAcc = self.videoName + '_' + str(self.userId) + 'Tk_EpoMax' + str(self.epochCNN) + '_' + \
                         str(self.epochRL) + '_AccAndBandwidth.csv'
        fileAcc = open(config.log_path + CSVfilenameAcc, 'w', newline='')
        CSVfilenameTime = self.videoName + '_' + str(self.userId) + 'Tk_EpoMax' + str(self.epochCNN) + '_' + \
                          str(self.epochRL) + '_TimeConsumption.csv'
        fileTime = open(config.log_path + CSVfilenameTime, 'w', newline='')

        self.writerAcc = csv.writer(fileAcc)
        self.writerTime = csv.writer(fileTime)

    """
    写入数据
    """
    def writerHead(self, TileNO):
        rows = [['VideoName', 'UserIndex', 'epochCNN', 'epochRL'],
                [self.videoName, str(self.userId), str(self.epochCNN), str(self.epochRL)]]
        self.writerAcc.writerows(rows)
        self.writerTime.writerows(rows)

        TotalSize = 'Size/' + str(TileNO * TileNO)
        sortedValues = ['accuracy', 'precision', 'recall', 'predicted tile', 'total size']
        self.writerAcc.writerow(sortedValues)

        ValueTime = ['running time', 'epochs', 'FirstLoss', 'FinalLoss']
        self.writerTime.writerow(ValueTime)

    """
    写入数据
    """

    def writerVideo(self, FrameRate, W_Frame, H_Frame):
        videoName = self.videoName + '_' + str(self.userId) + 'Tk_EpoMax' + str(self.epochCNN) + '_' + \
        str(self.epochRL) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.log_path + videoName, fourcc, FrameRate, (int(W_Frame), int(H_Frame)))
        return out

    def writerCSVT(self, content):
        self.writerTime.writerow(content)

    def writerCSVA(self, content):
        self.writerTime.writerow(content)
