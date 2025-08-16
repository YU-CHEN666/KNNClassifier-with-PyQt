from PyQt6 import QtCore,QtGui,QtWidgets
import os,numpy,pandas,pickle,shutil,sqlite3,pymysql,re
from pymysql.err import OperationalError
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier as KNClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import cross_val_score,StratifiedKFold



#選擇檔案視窗
class Ui_chose_file(QtWidgets.QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.setWindowTitle("選取檔案")
        self.setFixedSize(571,434)
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.text_show = QtWidgets.QTextEdit(parent=self)  
        self.text_show.setGeometry(QtCore.QRect(20, 20, 531, 341))
        self.text_show.setReadOnly(True)
        self.text_show.keyPressEvent = self.ctrl_c_pass
        self.text_show.contextMenuEvent = self.contextMenu_pass
        self.horizontalLayoutWidget = QtWidgets.QWidget(parent=self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 370, 531, 61))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setSpacing(3)
        #選則檔案按鈕
        self.button_chose_file = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.button_chose_file.setText("選擇檔案")
        self.button_chose_file.setStyleSheet("""height:46px;
                                             margin:0px 21px;
                                             font-size:25px;
                                             """)
        self.button_chose_file.clicked.connect(self.openfile)
        self.horizontalLayout.addWidget(self.button_chose_file)
        #從SQL按鈕
        self.button_chose_sql = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.button_chose_sql.setText("從SQL匯入")
        self.button_chose_sql.setStyleSheet("""height:46px;
                                             margin:0px 21px;
                                             font-size:23px;
                                             """)
        self.button_chose_sql.clicked.connect(lambda:sql_chose.show_set_activate())
        self.horizontalLayout.addWidget(self.button_chose_sql)
        #取消按鈕
        self.button_cancel = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.button_cancel.setText("取消")
        self.button_cancel.setStyleSheet("""height:46px;
                                         margin:0px 21px;
                                         font-size:25px;
                                         """)
        self.button_cancel.clicked.connect(lambda:self.close())
        self.horizontalLayout.addWidget(self.button_cancel)


    def format_error_dialog(self,file_name,where):
        #格式錯誤通知視窗
        wrong = QtWidgets.QMessageBox(self)
        wrong.setWindowTitle("格式錯誤")
        wrong.setStyleSheet("""font-size:20px;
                            QPushButton{background:white;
                                        font-size:8px}""")
        wrong.setStandardButtons(wrong.StandardButton.Ok)
        wrong.setIcon(wrong.Icon.Critical)
        if where == "特徵":
            wrong.setInformativeText(f"檔案名稱:{file_name}\n特徵資料中包含非數字資料，請修改為數字重新儲存，再重新選擇檔案。")
        elif where == "SQL特徵":
            wrong.setInformativeText(f"資料表名稱:{file_name}\n特徵資料中包含非數字資料，請修改為數字重新儲存，再重新選擇資料表。")
        elif where == "類別":
            wrong.setInformativeText(f"檔案名稱:{file_name}\n類別資料中包含非文字資料，請修改為文字重新儲存，再重新選擇檔案。")
        elif where == "SQL類別":
            wrong.setInformativeText(f"資料表名稱:{file_name}\n類別資料中包含非文字資料，請修改為文字重新儲存，再重新選擇資料表。")
        elif where == "合併":
            wrong.setInformativeText("選取的檔案特徵數量(欄位數量)不相同，請檢查修改後再重新選擇檔案。")
        elif where == "無模型":
            wrong.setWindowTitle("發生錯誤")
            wrong.setInformativeText("偵測到無任何模型存在，請先訓練模型。")
        elif where == "預測特徵數量":
            wrong.setInformativeText("待預測資料的特徵數量(欄位數量)，與模型不相同無法進行預測。")
        elif where == "確認刪除全部模型":
            wrong.setIcon(wrong.Icon.Question)
            wrong.setStandardButtons(wrong.StandardButton.Yes|wrong.StandardButton.No)
            wrong.setInformativeText("一旦開始便無法停止也無法復原，確定要繼續嗎?")
            wrong.setWindowTitle("確認全部刪除")
        elif where == "刪除特定模型":
            wrong.setIcon(wrong.Icon.Information)
            wrong.setWindowTitle("通知")
            wrong.setInformativeText("刪除成功!!!")
        elif where == "未選擇模型":
            wrong.setWindowTitle("發生錯誤")
            wrong.setInformativeText("請先選擇模型。")
        elif where == "未輸入":
            wrong.setWindowTitle("發生錯誤")
            wrong.setInformativeText("有欄位未輸入，請全部輸入完再按確認!")
        elif where == "主機位址":
            wrong.setWindowTitle("連線錯誤")
            wrong.setInformativeText("主機位址錯誤!!!無法連線。")
        elif where == "連接埠":
            wrong.setWindowTitle("連線錯誤")
            wrong.setInformativeText("連接埠錯誤!!!或是資料庫未啟動無法連線。")
        elif where == "資料庫名稱":
            wrong.setWindowTitle("連線錯誤")
            wrong.setInformativeText("資料庫不存在!!!無法連線。")
        elif where == "名稱和密碼":
            wrong.setWindowTitle("連線錯誤")
            wrong.setInformativeText("使用者名稱或密碼錯誤!!!無法連線。")
        return wrong.exec()


    def format_check(self,file_name,file_path,csv):
        #資料格式檢查
        if self.process.wasCanceled():
            return None,None
        if self.model == "pred":
            if len(csv.shape) == 1:
                csv = csv[numpy.newaxis,:]            
            if numpy.isnan(csv[0,0]):
                #有欄位名稱
                csv_data = csv[1:,:]
            else:
                csv_data = csv
            with open("Feature_number/"+default_model+"-number.pkl","rb") as f:
                pred_feature_number = pickle.load(f)
            if csv_data.shape[1] != pred_feature_number:
                self.process.cancel()
                self.format_error_dialog("","預測特徵數量")
                return None,None
            if numpy.any(numpy.isnan(csv_data)):
                self.process.cancel()
                self.format_error_dialog(file_name,"特徵")
                return None,None      
            return csv_data,None
        else:            
            if numpy.isnan(csv[0,0]):            
                #有欄位名稱            
                csv_data = csv[1:,:-1]
                csv_class = numpy.genfromtxt(file_path,dtype="str",delimiter=',',usecols=(-1),skip_header=1,encoding="utf-8")
            else:
                csv_data = csv[:,:-1]
                csv_class = numpy.genfromtxt(file_path,dtype="str",delimiter=',',usecols=(-1),encoding="utf-8")               
            if numpy.any(numpy.isnan(csv_data)):
                self.process.cancel()
                self.format_error_dialog(file_name,"特徵")
                return None,None
            elif  not numpy.all(numpy.isnan(csv[:,-1])):
                self.process.cancel()
                self.format_error_dialog(file_name,"類別")
                return None,None
            return csv_data,csv_class


    @staticmethod
    def class_coding(csv_class):
        #字串編碼成整數
        encoder = LabelEncoder()
        encoder.fit(csv_class)
        return encoder,encoder.transform(csv_class)
     
      
    def train_model(self,feature,target):
        #訓練模型
        scaler = StandardScaler()
        if self.process.wasCanceled():
            return None,None
        scaler.fit(feature)
        train_feature = scaler.transform(feature)
        train_number = feature.shape[0]
        k_value = int(train_number**0.5)
        #尋找最佳K值
        if k_value<=4:
            kn = KNClassifier(3)
            kn.fit(train_feature,target)
            return kn,scaler,train_feature.shape[1]
        else:
            while(k_value%2 == 0):
                k_value-=1
            test_k_value = [k_value-2,k_value,k_value+2] 
        score_list = []                                  
        for i in test_k_value:
            score = cross_val_score(KNClassifier(i),train_feature,target,cv=StratifiedKFold())            
            score_list.append(numpy.mean(score))
        best_k_value = test_k_value[score_list.index(max(score_list))]
        kn = KNClassifier(best_k_value)
        kn.fit(train_feature,target)
        return kn,scaler,train_feature.shape[1]


    @staticmethod
    def model_scaler_encoder_number_save(model,scaler,encoder,feature_number):
        #儲存模型&標準化器
        global default_model
        time_now = str(datetime.today().strftime("%Y-%m-%d %H.%M.%S"))
        for _ in ["Model","Scaler","Encoder","Feature_number"]:
            if os.path.exists(_):
                if _ == "Model":
                    with open("Model/"+time_now+"-model.pkl","wb") as f:
                        pickle.dump(model,f)
                elif _ == "Scaler":
                    with open("Scaler/"+time_now+"-scaler.pkl","wb") as f:
                        pickle.dump(scaler,f)
                elif _ == "Encoder":
                    with open("Encoder/"+time_now+"-encoder.pkl","wb") as f:
                        pickle.dump(encoder,f)
                else:
                    with open("Feature_number/"+time_now+"-number.pkl","wb") as f:
                        pickle.dump(feature_number,f)
            else:
                os.mkdir(_)
                if _ == "Model":
                    with open("Model/"+time_now+"-model.pkl","wb") as f:
                        pickle.dump(model,f)
                elif _ == "Scaler":
                    with open("Scaler/"+time_now+"-scaler.pkl","wb") as f:
                        pickle.dump(scaler,f)
                elif _ == "Encoder":
                    with open("Encoder/"+time_now+"-encoder.pkl","wb") as f:
                        pickle.dump(encoder,f)
                else:
                    with open("Feature_number/"+time_now+"-number.pkl","wb") as f:
                        pickle.dump(feature_number,f)
        with open("latest-date.pkl","wb") as f:
            pickle.dump(time_now,f)
        mainwindows.model_show_content.setText(time_now)
        default_model = time_now
        
                                
    def openfile(self):
        #選擇檔案對話視窗        
        if "USERPROFILE" in os.environ:            
            files,_ = QtWidgets.QFileDialog.getOpenFileNames(self,"開啟檔案",os.environ["USERPROFILE"]+"\\Desktop","(*.csv *.xlsx *.xls)")
        else:
            files,_ = QtWidgets.QFileDialog.getOpenFileNames(self,"開啟檔案",filter="(*.csv *.xlsx *.xls)")
        if files:
            #有選擇檔案
            self.process = QtWidgets.QProgressDialog(self)
            self.process.setAutoClose(False)
            self.process.setAutoReset(False)
            self.process.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            self.process.setWindowTitle("目前進度")
            self.process.setStyleSheet(processstytle)
            self.process.setMinimumSize(QtCore.QSize(300,110))
            self.process_value = 0
            self.process.setValue(self.process_value)            
            self.process.canceled.connect(self.process_canel)
            self.process.setLabelText("處理資料中......")
            self.process.setMinimumDuration(1)
            csv_data,csv_class = self.get_data_class_process(files)
            if csv_data is None:
                self.process.cancel()
                return                                   
            elif self.model == "train":
                self.process.setLabelText("訓練模型中......")
                self.process_value = 75
                self.process.setValue(self.process_value)                
                encoder,train_class = self.class_coding(csv_class)
                model,scaler,feature_number = self.train_model(csv_data,train_class)
                if model is None:
                    self.process.cancel()
                    return
                self.model_scaler_encoder_number_save(model,scaler,encoder,feature_number)
                self.process.setLabelText("訓練完成")
                self.process_value = 100
                self.process.setValue(self.process_value)               
            else:
                self.process.setLabelText("模型預測中......")
                self.process_value = 75
                self.process.setValue(self.process_value)
                with open("./Model/"+default_model+"-model.pkl","rb") as f:
                    pred_model = pickle.load(f)
                with open("./Scaler/"+default_model+"-scaler.pkl","rb") as f:
                    pred_model_scaler = pickle.load(f)
                with open("./Encoder/"+default_model+"-encoder.pkl","rb") as f:
                    pred_model_encoder = pickle.load(f)
                pred_data = pred_model_scaler.transform(csv_data)
                result = pred_model.predict(pred_data)
                if len(result)==1:
                    #只有一筆預測資料
                    result = pred_model_encoder.inverse_transform(result)
                    mainwindows.pred_show_content.setText(str(result[0]))
                    self.process.setLabelText("預測完成")
                    self.process_value = 100
                    self.process.setValue(self.process_value)
                else:
                    result = pred_model_encoder.inverse_transform(result)
                    result = result[:,numpy.newaxis]
                    if "USERPROFILE" in os.environ:
                        numpy.savetxt(os.environ["USERPROFILE"]+"\\Desktop\\result.txt",result,"%s")
                        mainwindows.pred_show_content.setText("已將結果儲存在桌面的result.txt檔案")
                    else:
                        numpy.savetxt(os.environ["SYSTEMDRIVE"]+"\\result.txt",result,"%s")
                        mainwindows.pred_show_content.setText(f"已將結果儲存在{os.environ['SYSTEMDRIVE']}中的result.txt檔案")
                    self.process.setLabelText("預測完成")
                    self.process_value = 100
                    self.process.setValue(self.process_value)
        else:
            return
  
    
    def excel_to_csv(self,path):
        #將excel轉換成csv
        excel =  pandas.read_excel(path)
        excel.to_csv("temp.csv",encoding="utf-8",index=False)
        csv = numpy.genfromtxt("temp.csv",dtype="float32",delimiter=',',encoding="utf-8")       
        return csv

    
    def merge_data_target(self,data_temp,class_temp,data,target,feature_numer):
        #合併特徵&類別資料
        if data.shape[1] != feature_numer:      
            return None,None
        if self.model == "train":           
            return numpy.vstack((data_temp,data)),numpy.hstack((class_temp,target))
        else:
            return numpy.vstack((data_temp,data)),None           

        
    def get_data_class_process(self,files):
        if len(files)==1:           
            #只選擇一個檔案
            _ = files[0].split("/")[-1].split(".")
            file_name = _[0]
            file_type = _[1]
            self.process_value = 10
            self.process.setValue(self.process_value)
            if "csv" in file_type:
                csv = numpy.genfromtxt(files[0],dtype="float32",delimiter=',',encoding="utf-8")
                csv_data,csv_class = self.format_check(file_name,files[0],csv)
                if csv_data is None:
                    return None,None
                self.process_value = 50
                self.process.setValue(self.process_value)
                return csv_data,csv_class
            else:
                csv = self.excel_to_csv(files[0])
                csv_data,csv_class = self.format_check(file_name,"temp.csv",csv)
                if csv_data is None:
                    os.remove("temp.csv")
                    return None,None
                os.remove("temp.csv")
                self.process_value = 50
                self.process.setValue(self.process_value)
                return csv_data,csv_class
        else:
            #選擇多個檔案
            _ = files[0].split("/")[-1].split(".")
            file_name = _[0]
            file_type = _[1]
            if "csv" in file_type:
                csv = numpy.genfromtxt(files[0],dtype="float32",delimiter=',',encoding="utf-8")
                csv_data_temp,csv_class_temp = self.format_check(file_name,files[0],csv)
                if csv_data_temp is None:
                    return None,None
                feature_numer = csv_data_temp.shape[1]
            else:
                csv = self.excel_to_csv(files[0])
                csv_data_temp,csv_class_temp = self.format_check(file_name,"temp.csv",csv)
                if csv_data_temp is None:
                    os.remove("temp.csv")
                    return None,None
                feature_numer = csv_data_temp.shape[1]
                os.remove("temp.csv")
            self.process_value = 10
            self.process.setValue(self.process_value)
            for file_path in files[1:]:
                if self.process.wasCanceled():
                    return None,None
                elif len(files[1:]) >1:
                    if files[1:].index(file_path) == len(files[1:])//2:
                        self.process_value = 30
                        self.process.setValue(self.process_value)
                _ = file_path.split("/")[-1].split(".")
                file_name = _[0]
                file_type = _[1]
                if "csv" in file_type:
                    csv = numpy.genfromtxt(file_path,dtype="float32",delimiter=',',encoding="utf-8")                    
                    csv_data,csv_class = self.format_check(file_name,file_path,csv)
                    if csv_data is None:
                        return None,None
                    csv_data_temp,csv_class_temp = self.merge_data_target(csv_data_temp,csv_class_temp,csv_data,csv_class,feature_numer)
                    if csv_data_temp is None:
                        self.process.cancel() 
                        self.format_error_dialog("","合併")
                        return None,None
                elif "xlsx" in file_type or "xls" in file_type:
                    csv = self.excel_to_csv(file_path)
                    csv_data,csv_class = self.format_check(file_name,"temp.csv",csv)
                    if csv_data is None:
                        os.remove("temp.csv")
                        return None,None
                    os.remove("temp.csv")
                    csv_data_temp,csv_class_temp = self.merge_data_target(csv_data_temp,csv_class_temp,csv_data,csv_class,feature_numer)
                    if csv_data_temp is None:
                        self.process.cancel()      
                        self.format_error_dialog("","合併")
                        return None,None
            self.process_value = 50
            self.process.setValue(self.process_value)
            return csv_data_temp,csv_class_temp
 
    
    def process_canel(self):
        if self.process_value<100:
            self.process.setValue(self.process_value)
            self.process.setLabelText("正在終止程序......")
            self.process.show()
      
    def ctrl_c_pass(self,event):
        combin = event.keyCombination()
        if combin.toCombined() == 67108931:
            return
        QtWidgets.QTextEdit.keyPressEvent(self.text_show,event)
 
    
    def contextMenu_pass(self,event):
        pass

   
    def whocall(self,who):
        #判斷主視窗中，是哪顆按鈕被按下(開始預測or訓練新模型)，並顯示對應的訊息
        global default_model
        train_str="注意事項:\n1.檔案副檔名只接受.csv、.xlsx、.xls。\n2.特徵值只接受正整數可以有小數點。\n3.可一次選擇多個檔案，但是特徵數量(欄位總數)要相同。SQL只能選擇一個資料表。\n4.特徵對應的種類值，必須緊鄰於特徵的最後，不可位於特徵之間。\n範例:"
        pred_str = "注意事項:\n1.檔案副檔名只接受.csv、.xlsx、.xls。\n2.如果有增加新特徵(欄位)，請先重新訓練模型，否則無法進行預測。\n3.可以選擇多個檔案，但要確保每個檔案的特徵數量，和訓練模型時的數量一樣。SQL只能選擇一個資料表。\n4.只預測一筆資料，結果會顯示在主頁面;預測多筆資料會儲存成txt檔案，每一行就是一筆預測結果，檔案位置資訊會顯示在主頁面。"
        self.text_show.clear()
        if who == "訓練":
            font = self.text_show.currentFont()
            font.setPointSize(16)
            self.text_show.setCurrentFont(font)            
            cursor = self.text_show.textCursor()
            cursor.insertText(train_str)
            cursor.insertBlock()
            image = QtGui.QImage("C:\\Users\\ksd55\\Desktop\\範例圖片.jpg")
            cursor.insertImage(image.scaledToWidth(650,QtCore.Qt.TransformationMode.SmoothTransformation))            
            cursor.movePosition(cursor.MoveOperation.Start)
            self.text_show.setTextCursor(cursor)
            self.model = "train"
        elif who == "預測":
            try:
                next(os.scandir("Model"))
            except StopIteration:
                Ui_chose_file.format_error_dialog(mainwindows,"","無模型")
                return
            else:
                if default_model == "未選擇模型":
                    Ui_chose_file.format_error_dialog(mainwindows,"","未選擇模型")
                    return 
                font = self.text_show.currentFont()
                font.setPointSize(16)
                self.text_show.setCurrentFont(font)
                cursor = self.text_show.textCursor()
                cursor.insertText(pred_str)
                cursor.insertBlock()
                cursor.movePosition(cursor.MoveOperation.Start)
                self.text_show.setTextCursor(cursor)
                self.model = "pred"
        self.show()


#MySQL連線參數輸入視窗
class Ui_mysql_chose(QtWidgets.QDialog):
    def __init__(self,parent):
        super().__init__(parent)
        self.setFixedSize(400,305)
        self.setWindowTitle("資料庫連線參數設定")
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        #輸入layout
        self.FormLayout_widget = QtWidgets.QWidget(self)
        self.FormLayout_widget.setGeometry(20,10,360,200)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.FormLayout_widget.setFont(font)
        self.input_layout = QtWidgets.QFormLayout(self.FormLayout_widget)
        for i in ["主機位址:","連接埠:","字元集:","資料庫名稱:","使用者名稱:","密碼:"]:            
            if i == "主機位址:":
                self.input_layout.addRow(i,QtWidgets.QLineEdit("localhost",self.FormLayout_widget))
            elif i == "連接埠:":
                self.input_layout.addRow(i,QtWidgets.QLineEdit("3306",self.FormLayout_widget))
            elif i == "字元集:":
                self.input_layout.addRow(i,QtWidgets.QLineEdit("utf8mb4",self.FormLayout_widget))
            elif i == "使用者名稱:":
                self.input_layout.addRow(i,QtWidgets.QLineEdit("root",self.FormLayout_widget))
            else:
                self.input_layout.addRow(i,QtWidgets.QLineEdit(self.FormLayout_widget))
        self.input_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.password_QLineEdit = self.input_layout.itemAt(5,self.input_layout.ItemRole.FieldRole).widget()       
        #顯示密碼選項
        self.password_show = QtWidgets.QRadioButton("顯示密碼",self)
        self.password_show.setFont(font)
        self.password_show.move(92,200)
        self.password_show.toggled.connect(self.set_password_show)
        #確認按鈕
        self.confirm_button = QtWidgets.QPushButton("確認",self)
        self.confirm_button.move(200,240)
        self.confirm_button.setStyleSheet("""height:40px; width:55px; font-size:23px;""")
        self.confirm_button.clicked.connect(self.check_input)
        #取消按鈕
        self.cancel_button = QtWidgets.QPushButton("取消",self)
        self.cancel_button.move(295,240)
        self.cancel_button.setStyleSheet("""height:40px; width:55px; font-size:23px;""")
        self.cancel_button.clicked.connect(lambda:self.done(0))
        #全部清除按鈕
        self.clear_button = QtWidgets.QPushButton("全部清除",self)
        self.clear_button.move(70,240)
        self.clear_button.setStyleSheet("""height:40px; width:100px; font-size:23px;""")
        self.clear_button.clicked.connect(self.clear_all)
        
    
    def set_password_show(self):
        #顯示密碼
        if self.password_show.isChecked():
            self.password_QLineEdit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
        else:
            self.password_QLineEdit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)

        
    def clear_all(self):
        #清空所有欄位
        for i in  range(self.input_layout.rowCount()):
            _ = self.input_layout.itemAt(i,self.input_layout.ItemRole.FieldRole).widget()
            _.setText("")
    
       
    def check_input(self):
        #檢查所有欄位是否都有輸入
        for i in range(self.input_layout.rowCount()):
            QLineEdit = self.input_layout.itemAt(i,self.input_layout.ItemRole.FieldRole).widget()
            if QLineEdit.text() == "":
                Ui_chose_file.format_error_dialog(self,"","未輸入")
                return
        self.done(1)
                
    
    def get_return(self):
        #顯示輸入視窗&傳回輸入資料
        self.password_QLineEdit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.password_show.setChecked(False)
        chose_state = self.exec()
        if chose_state == 1:
            _ = ["host","port","charset","db","user","passwd"]
            data = {}
            for i in range(self.input_layout.rowCount()):
               QLineEdit = self.input_layout.itemAt(i,self.input_layout.ItemRole.FieldRole).widget()
               if i == 1:
                   data[_[i]]=int(QLineEdit.text())
               else:
                   data[_[i]]=QLineEdit.text()
            return data
        else:
            return None
            
                                      
        
#SQL選擇視窗
class Ui_sql_chose(QtWidgets.QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.setFixedSize(260,80)
        self.setWindowTitle("資料庫類型選擇")
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        font = QtGui.QFont()
        font.setPointSize(14)
        #SQLite按鈕
        self.sqlite_button = QtWidgets.QPushButton("SQLite",self)
        self.sqlite_button.setGeometry(20,10,100,50)
        self.sqlite_button.setFont(font)
        self.sqlite_button.clicked.connect(self.sqlite_process)
        #MySQL按鈕
        self.mysql_button = QtWidgets.QPushButton("MySQL",self)
        self.mysql_button.setGeometry(140,10,100,50)
        self.mysql_button.setFont(font)
        self.mysql_button.clicked.connect(self.mysql_process)
        self.isactivate = False
        
        
    def show_set_activate(self):
        #顯示SQL選擇視窗並且設定為處於活動狀態
        self.isactivate = True
        self.show()
        
    
    def format_error_dialog(self,table_name,where):
        if where == "特徵":            
            chose_file.format_error_dialog(table_name,"SQL特徵")
        elif where == "類別":
            chose_file.format_error_dialog(table_name,"SQL類別")
        else:
            chose_file.format_error_dialog("","預測特徵數量")
     
        
    def create_processdialog(self):
        #建立進度條
        self.process = QtWidgets.QProgressDialog(self)
        self.process.setAutoClose(False)
        self.process.setAutoReset(False)
        self.process.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.process.setWindowTitle("目前進度")
        self.process.setStyleSheet(processstytle)
        self.process.setMinimumSize(QtCore.QSize(300,110))
        self.process_value = 0
        self.process.setValue(self.process_value)            
        self.process.canceled.connect(self.process_canel)
        self.process.setLabelText("處理資料中......")
        self.process.setMinimumDuration(1)


    def process_canel(self):
        if self.process_value<100:
            self.process.setValue(self.process_value)
            self.process.setLabelText("正在終止程序......")
            self.process.show()

            
    def sqldata_format_check(self,data,source_data,table_name):
        #資料格式檢查，專為從SQL讀取的資料設計
        if self.process.wasCanceled():
            return None,None
        if chose_file.model == "train":
            feature = data[:,:-1]
            target = data[:,-1]
            if numpy.any(numpy.isnan(feature)):
                self.process.cancel()
                self.format_error_dialog(table_name,"特徵")
                return None,None
            if not numpy.all(numpy.isnan(target)):
                self.process.cancel()
                self.format_error_dialog(table_name,"類別")
                return None,None
            self.process_value = 50
            self.process.setValue(self.process_value)
            return feature,source_data[:,-1]            
        else:
            with open("Feature_number/"+default_model+"-number.pkl","rb") as f:
                pred_feature_number = pickle.load(f)
            if data.shape[1] != pred_feature_number:
                self.process.cancel()
                self.format_error_dialog("","預測特徵數量")
                return None,None           
            if numpy.any(numpy.isnan(data)):
                self.process.cancel()
                self.format_error_dialog(table_name,"特徵")
                return None,None
            self.process_value = 50
            self.process.setValue(self.process_value)
            return data,None     

    
    def mysql_process(self):
        self.close()
        connect_parameters = mysql_input.get_return()
        if connect_parameters is None:
            return
        else:
            try:
                mysql = pymysql.connect(**connect_parameters)
            except OperationalError as e:
                code,message = e.args
                host_error = re.compile("11001")
                if code==2003:
                    _ = host_error.search(message)
                    if _ is None:
                        chose_file.format_error_dialog("","連接埠")
                    else:
                        chose_file.format_error_dialog("","主機位址")
                elif code==1049:
                    chose_file.format_error_dialog("","資料庫名稱")
                elif code==1045:
                    chose_file.format_error_dialog("","名稱和密碼")                    
            else:
                with mysql:
                    with mysql.cursor() as cursor:
                        cursor.execute("SHOW TABLES")
                        _ = cursor.fetchall()
                        res = [i[0] for i in _]
                        if len(res)>1:
                            chose,chose_state = chose_model_year.get_return(res)
                            if not chose_state:
                                self.isactivate = False
                                return
                            res = chose
                        else:
                            res = res[0]
                        cursor.execute(f"SELECT * FROM `{res}`")
                        search_result = cursor.fetchall()
                        self.data_process(search_result,res)
 
                                                        
    def sqlite_process(self):
        #SQLite-資料庫&資料表選擇
        self.close()
        if "USERPROFILE" in os.environ:            
            file,_ = QtWidgets.QFileDialog.getOpenFileName(self,"開啟資料庫",os.environ["USERPROFILE"]+"\\Desktop","(*.db *.db3 *.sqlite *.sqlite3)")
        else:
            file,_ = QtWidgets.QFileDialog.getOpenFileName(self,"開啟資料庫",filter="(*.db *.db3 *.sqlite *.sqlite3)")
        if file:
            sql = sqlite3.connect(file)
            _ = sql.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            res = [i[0] for i in _]
            if len(res)>1:
                chose,chose_state = chose_model_year.get_return(res)
                if not chose_state:
                    self.isactivate = False
                    return
                res = chose
            else:
                res = res[0]
            search_result = sql.execute(f"SELECT * FROM '{res}'").fetchall()            
            self.data_process(search_result,res)
            sql.close()
        else:
            return            
    
    
    def data_process(self,data_from_table,table_name):
        #資料處裡for預測,訓練
        self.create_processdialog()
        initial_data = numpy.array(data_from_table,dtype="str")
        initial_data_shape = initial_data.shape
        data = pandas.to_numeric(initial_data.reshape(-1),errors='coerce',downcast="float")
        data = data.reshape(initial_data_shape)
        if chose_file.model == "train":
            feature,target = self.sqldata_format_check(data,initial_data,table_name)
            if feature is None:
                self.process.cancel()
                return
            encoder,train_class = Ui_chose_file.class_coding(target)
            if self.process.wasCanceled():
                self.process.cancel()
                return
            self.process_value = 75
            self.process.setValue(self.process_value)
            self.process.setLabelText("訓練模型中......")
            model,scaler,feature_number = Ui_chose_file.train_model(self,feature,train_class)
            if model is None:
                self.process.cancel()
                return
            Ui_chose_file.model_scaler_encoder_number_save(model,scaler,encoder,feature_number)
            self.process_value = 100
            self.process.setValue(self.process_value)
            self.process.setLabelText("訓練完成")               
        else:
            feature,_ = self.sqldata_format_check(data,None,None)
            if feature is None:
                self.process.cancel()
                return
            with open("./Model/"+default_model+"-model.pkl","rb") as f:
                pred_model = pickle.load(f)
            with open("./Scaler/"+default_model+"-scaler.pkl","rb") as f:
                pred_model_scaler = pickle.load(f)
            with open("./Encoder/"+default_model+"-encoder.pkl","rb") as f:
                pred_model_encoder = pickle.load(f)
            self.process.setLabelText("模型預測中......")
            self.process_value = 75
            self.process.setValue(self.process_value)               
            pred_data = pred_model_scaler.transform(feature)
            result = pred_model.predict(pred_data)
            if len(result)==1:
                #只有一筆預測資料
                result = pred_model_encoder.inverse_transform(result)
                mainwindows.pred_show_content.setText(str(result[0]))
            else:
                result = pred_model_encoder.inverse_transform(result)
                result = result[:,numpy.newaxis]
                if "USERPROFILE" in os.environ:
                    numpy.savetxt(os.environ["USERPROFILE"]+"\\Desktop\\result.txt",result,"%s")
                    mainwindows.pred_show_content.setText("已將結果儲存在桌面的result.txt檔案")
                else:
                    numpy.savetxt(os.environ["SYSTEMDRIVE"]+"\\result.txt",result,"%s")
                    mainwindows.pred_show_content.setText(f"已將結果儲存在{os.environ['SYSTEMDRIVE']}中的result.txt檔案")
            self.process.setLabelText("預測完成")
            self.process_value = 100
            self.process.setValue(self.process_value)
        self.isactivate = False

        
       
#選擇模型視窗-年份
class Ui_chose_model_year(QtWidgets.QDialog):
    def __init__(self,parent):
        super().__init__(parent)
        self.setFixedSize(280,130)
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        font = QtGui.QFont()
        font.setPointSize(13)
        #內容標題
        self.title = QtWidgets.QLabel(self)
        self.title.setGeometry(20,10,120,30)
        self.title.setFont(font)
        #選單
        self.box = QtWidgets.QComboBox(self)
        self.box.setGeometry(20,45,240,35)
        self.box.setFont(font)
        #確認按鈕
        self.ok_button = QtWidgets.QPushButton("確認",self)
        self.ok_button.setGeometry(100,90,75,30)
        self.ok_button.setFont(font)
        self.ok_button.clicked.connect(self.after_chose)
        #取消按鈕
        self.cancel_button = QtWidgets.QPushButton("取消",self)
        self.cancel_button.setGeometry(185,90,75,30)
        self.cancel_button.setFont(font)
        self.cancel_button.clicked.connect(self.afer_cancel)
 
        
    def add_items_show(self,Iterable,who):
        self.box.clear()
        self.box.addItems(Iterable)
        self.setWindowTitle("選擇年份")
        self.title.setText("請選擇年份")
        if who == "delete":
            delete_model.close()
        chose_model_final.who = who
        self.show()
            
    
    def after_chose(self):        
        self.close()
        if sql_chose.isactivate:
            self.done(1)
        else:
            chose_model_final.add_items_show(self.box.currentText())
    
    
    def afer_cancel(self):
        if sql_chose.isactivate:
            self.done(0)
        else:
            self.close()
 
    
    def get_return(self,Iterable):
        self.box.clear()
        self.box.addItems(Iterable)
        self.setWindowTitle("選擇資料表")
        self.title.setText("請選擇資料表")
        chose_state = self.exec()
        if chose_state == 1:
            return self.box.currentText(),True
        else:
            return None,False
 
    
        
#選擇模型視窗-最終
class Ui_chose_model_final(QtWidgets.QWidget):   
    def __init__(self,parent):
        super().__init__(parent)
        self.setFixedSize(260,220)
        self.setWindowTitle("選擇模型")
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        font = QtGui.QFont()
        font.setPointSize(13)
        #內容標題
        self.title = QtWidgets.QLabel("請選擇模型",self)
        self.title.setGeometry(20,10,90,30)
        self.title.setFont(font)
        #清單
        self.List = QtWidgets.QListWidget(self)
        self.List.setGeometry(20,45,220,130)
        font_list = QtGui.QFont()
        font_list.setPointSize(15)
        self.List.setFont(font_list)
        self.List.setStyleSheet("""QListWidget::item:selected{background:blue;}""")
        #確認按鈕
        self.ok_button = QtWidgets.QPushButton("確認",self)
        self.ok_button.setGeometry(80,180,75,30)
        self.ok_button.setFont(font)
        self.ok_button.clicked.connect(self.after_chose)
        #取消按鈕
        self.cancel_button = QtWidgets.QPushButton("取消",self)
        self.cancel_button.setGeometry(165,180,75,30)
        self.cancel_button.setFont(font)
        self.cancel_button.clicked.connect(lambda:self.close())

    
    def add_items_show(self,chose):
        self.List.clear()
        self.List.addItems(model_year[chose])
        self.show()

    
    def after_chose(self,who):
        global model_year
        global default_model
        global current_model_number
        if self.List.selectedItems():
            self.close()
            if self.who == "pred":
                default_model = self.List.currentItem().text()
                mainwindows.model_show_content.setText(default_model)
            else:
                model_chosed = self.List.currentItem().text()
                os.remove("Model/"+model_chosed+"-model.pkl")
                os.remove("Scaler/"+model_chosed+"-scaler.pkl")
                os.remove("Encoder/"+model_chosed+"-encoder.pkl")
                os.remove("Feature_number/"+model_chosed+"-number.pkl")
                years_list = list(model_year.keys())
                latest_model = model_year[years_list[-1]][-1]
                if current_model_number == 1:
                    mainwindows.model_show_content.setText("未訓練模型")
                    default_model = ""
                    with open("latest-date.pkl","wb") as f:                        
                        pickle.dump("",f)
                else:
                    if model_chosed == latest_model:
                        _ = find_second_latest_model(years_list)                    
                        with open("latest-date.pkl","wb") as f:                        
                            pickle.dump(_,f)
                    if model_chosed == default_model:
                        default_model = "未選擇模型"
                        mainwindows.model_show_content.setText(default_model)                                    
                Ui_chose_file.format_error_dialog(mainwindows,"","刪除特定模型")
 


#刪除模型視窗
class Ui_delete_model(QtWidgets.QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.setFixedSize(260,80)
        self.setWindowTitle("模式選擇")
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        font = QtGui.QFont()
        font.setPointSize(14)
        #刪除特定按鈕
        self.specific_button = QtWidgets.QPushButton("刪除特定",self)
        self.specific_button.setGeometry(20,10,100,50)
        self.specific_button.setFont(font)
        self.specific_button.clicked.connect(lambda:mainwindows.start_chose_model("delete"))
        #刪除全部按鈕
        self.all_button = QtWidgets.QPushButton("刪除全部",self)
        self.all_button.setGeometry(140,10,100,50)
        self.all_button.setFont(font)
        self.all_button.clicked.connect(self.delete_all)

    
    def delete_all(self):
        self.close()
        answer = Ui_chose_file.format_error_dialog(mainwindows,"","確認刪除全部模型")
        global default_model
        if answer == 16384:            
            process = QtWidgets.QProgressDialog(mainwindows)
            button = QtWidgets.QPushButton("ok",process)
            process.setCancelButton(button)
            button.hide()
            process.setAutoClose(False)
            process.setAutoReset(False)
            process.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            process.setWindowTitle("目前進度")
            process.setStyleSheet(processstytle)
            process.setMinimumSize(QtCore.QSize(300,110))
            process.setValue(0)
            process.setLabelText("刪除模型中......")
            process.setMinimumDuration(1)
            shutil.rmtree("Model")
            os.mkdir("Model")
            process.setValue(20)
            shutil.rmtree("Scaler")
            os.mkdir("Scaler")
            process.setValue(40)
            shutil.rmtree("Encoder")
            os.mkdir("Encoder")
            process.setValue(60)
            shutil.rmtree("Feature_number")
            os.mkdir("Feature_number")
            process.setValue(80)
            default_model = ""
            mainwindows.model_show_content.setText("未訓練模型")
            with open("latest-date.pkl","wb") as f:
                default_model = pickle.dump("",f)
            process.setValue(100)
            process.setLabelText("刪除完成")
            button.show()
        else:
            return



#主視窗
class Ui_mainwindows(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(811, 402)
        self.setWindowTitle( "KNN分類器")
        #按鈕layout
        self.button_layout_Widget = QtWidgets.QWidget(parent=self)
        self.button_layout_Widget.setGeometry(QtCore.QRect(0, 0, 201, 401))
        self.button_layout = QtWidgets.QVBoxLayout(self.button_layout_Widget)
        self.button_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)
        self.button_layout.setSpacing(4)
        #開始預測按鈕
        self.start_pred_button = QtWidgets.QPushButton(self.button_layout_Widget)
        self.start_pred_button.setText("開始預測")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.start_pred_button.setFont(font)
        self.start_pred_button.setStyleSheet("height:60px")
        self.start_pred_button.clicked.connect(lambda:chose_file.whocall("預測"))
        self.button_layout.addWidget(self.start_pred_button)
        #訓練新模型按鈕
        self.train_new_model_button = QtWidgets.QPushButton(self.button_layout_Widget)
        self.train_new_model_button.setText("訓練新模型")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.train_new_model_button.setFont(font)
        self.train_new_model_button.setStyleSheet("height:60px")
        self.train_new_model_button.clicked.connect(lambda:chose_file.whocall("訓練"))
        self.button_layout.addWidget(self.train_new_model_button)
        #選擇現有按鈕
        self.chose_model_button = QtWidgets.QPushButton(self.button_layout_Widget)
        self.chose_model_button.setText("選擇模型")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.chose_model_button.setFont(font)
        self.chose_model_button.setStyleSheet("height:60px")
        self.chose_model_button.clicked.connect(lambda:self.check_exist_model("pred"))
        self.button_layout.addWidget(self.chose_model_button)
        #刪除模型
        self.delete_model_button = QtWidgets.QPushButton(self.button_layout_Widget)
        self.delete_model_button.setText("刪除模型")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.delete_model_button.setFont(font)
        self.delete_model_button.setStyleSheet("height:60px")
        self.delete_model_button.clicked.connect(lambda:self.check_exist_model("delete"))
        self.button_layout.addWidget(self.delete_model_button)
        #標題layout
        self.title_Layout_Widget = QtWidgets.QWidget(parent=self)
        self.title_Layout_Widget.setGeometry(QtCore.QRect(260, 100, 130, 181))
        self.title_Layout = QtWidgets.QVBoxLayout(self.title_Layout_Widget)
        #當前模型-標題
        self.model_show_title = QtWidgets.QLabel(self.title_Layout_Widget)
        self.model_show_title.setText("當前模型:")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.model_show_title.setFont(font)
        self.title_Layout.addWidget(self.model_show_title)
        #預測結果-標題
        self.pred_show_title = QtWidgets.QLabel(self.title_Layout_Widget)
        self.pred_show_title.setText("預測結果:")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pred_show_title.setFont(font)
        self.title_Layout.addWidget(self.pred_show_title)
        #內容layout
        self.show_content_Widget = QtWidgets.QWidget(parent=self)
        self.show_content_Widget.setGeometry(QtCore.QRect(400, 100, 381, 181))
        self.content_Layout = QtWidgets.QVBoxLayout(self.show_content_Widget)
        #當前模型-內容
        self.model_show_content = QtWidgets.QLabel(self.show_content_Widget)
        if default_model == "":
            self.model_show_content.setText("未訓練模型")
        else:
            self.model_show_content.setText(default_model)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.model_show_content.setFont(font)
        self.model_show_content.setStyleSheet("background:white")
        self.content_Layout.addWidget(self.model_show_content)
        #預測結果-內容
        self.pred_show_content = QtWidgets.QLabel(self.show_content_Widget)
        self.pred_show_content.setText("未預測")
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pred_show_content.setFont(font)
        self.pred_show_content.setStyleSheet("background:white")
        self.content_Layout.addWidget(self.pred_show_content)
    
    
    def check_exist_model(self,who):
        global default_model
        try:
            next(os.scandir("Model"))
        except StopIteration:
            Ui_chose_file.format_error_dialog(self,"","無模型")
            return
        else:
            if who == "pred":
                self.start_chose_model("pred")
            else:
                delete_model.show()
            
            
    def start_chose_model(self,who):
            global model_year
            global current_model_number
            model_year = {}
            current_model_number = 0
            with os.scandir("Model") as f:
                for file in f:
                    current_model_number+=1
                    year = file.name.split(" ")[0].split("-")[0]
                    file_name = file.name.split("-model")[0]
                    if not year in model_year:
                        model_year[year] = [file_name]
                    else:
                        model_year[year].append(file_name)
            chose_model_year.add_items_show(model_year.keys(),who)




def find_second_latest_model(year_list,latest_year="yes"):
    global model_year
    if len(model_year[year_list[-1]])==1 and latest_year=="yes":
        return find_second_latest_model(year_list[:-1],"no")
    if latest_year=="yes":
        return model_year[year_list[-1]][-2]
    else:
        return model_year[year_list[-1]][-1]

                                                         
with open("latest-date.pkl","rb") as f:
    default_model = pickle.load(f)


processstytle = """
QLabel{font-size:20px;}
QProgressBar{width:100px;
height:20px;
text-align:center;
font-size:15px;
border:1px solid gray;
border-radius:5px;}
QProgressBar::chunk{background:#66c2ff;}
QPushButton{width:70px; height:25px; font-size:15px;}"""


app = QtWidgets.QApplication([])
mainwindows = Ui_mainwindows()
chose_file = Ui_chose_file(mainwindows)
chose_model_year = Ui_chose_model_year(mainwindows)
chose_model_final = Ui_chose_model_final(mainwindows)
delete_model = Ui_delete_model(mainwindows)
sql_chose = Ui_sql_chose(mainwindows)
mysql_input = Ui_mysql_chose(mainwindows)
mainwindows.show()
app.exec()