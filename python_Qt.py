import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QMessageBox, QProgressBar, QFileDialog, QLabel, QScrollArea, QGridLayout
from PyQt5.QtCore import QProcess
from PyQt5.QtGui import QPixmap
from os import listdir
from os.path import isfile, join

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("训练与测试脚本")
        self.showMaximized()  # 设置窗口全屏显示

        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建布局
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 创建按钮
        self.train_button = QPushButton("训练")
        self.test_button = QPushButton("测试")
        # 创建中断按钮和日志按钮
        self.stop_button = QPushButton("中断操作")
        self.log_button = QPushButton("查看日志")
        # 创建显示图像按钮
        # 创建显示所有图像按钮
        self.show_all_images_button = QPushButton("显示所有检测结果图像")

        # 添加按钮到布局
        self.layout.addWidget(self.train_button)
        self.layout.addWidget(self.test_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.log_button)
        self.layout.addWidget(self.show_all_images_button)

        # 创建输出窗口
        self.output_window = QTextEdit()
        self.output_window.setReadOnly(True)
        self.layout.addWidget(self.output_window)

        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 设置为不确定进度
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        # 创建图像显示区域
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)  # 图像自适应标签大小
        self.layout.addWidget(self.image_label)

        # 创建滚动区域以显示多张图像
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        # 修改滚动区域布局为网格布局
        self.scroll_area_layout = QGridLayout()
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # 连接按钮点击事件
        self.train_button.clicked.connect(self.run_train_script)
        self.test_button.clicked.connect(self.run_test_script)
        # 连接按钮点击事件
        self.stop_button.clicked.connect(self.stop_process)
        self.log_button.clicked.connect(self.view_log)
        self.show_all_images_button.clicked.connect(self.show_all_images)

        # 创建进程
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

    def run_train_script(self):
        self.output_window.clear()
        self.progress_bar.setVisible(True)
        self.process.start("python", ["E:/model/EmbeddingAD-main/MVTec2.py"])

    def run_test_script(self):
        self.output_window.clear()
        self.progress_bar.setVisible(True)
        self.process.start("python", ["E:/model/EmbeddingAD-main/evaluate_mvtec.py"])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        try:
            text = data.data().decode("utf-8")
        except UnicodeDecodeError:
            text = data.data().decode("latin1")  # 使用备用编码解码
        self.output_window.append(text)
        self.output_window.ensureCursorVisible()

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        try:
            text = data.data().decode("utf-8")
        except UnicodeDecodeError:
            text = data.data().decode("latin1")  # 使用备用编码解码
        self.output_window.append(text)
        self.output_window.ensureCursorVisible()

    def process_finished(self):
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "完成", "脚本运行完成！")

    def stop_process(self):
        if self.process.state() == QProcess.Running:
            self.process.kill()
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "中断", "操作已被中断！")

    def view_log(self):
        log_path, _ = QFileDialog.getOpenFileName(self, "选择日志文件", "E:/model/EmbeddingAD-main", "日志文件 (*.log *.txt)")
        if log_path:
            with open(log_path, "r", encoding="utf-8") as log_file:
                log_content = log_file.read()
            self.output_window.clear()
            self.output_window.append(log_content)


    def show_all_images(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹", "E:/model/EmbeddingAD-main")
        if folder_path:
            # 清空滚动区域
            for i in reversed(range(self.scroll_area_layout.count())):
                widget = self.scroll_area_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

            # 加载文件夹中的所有图像
            image_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            row, col = 0, 0
            for image_file in image_files:
                image_path = join(folder_path, image_file)
                pixmap = QPixmap(image_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.adjustSize()  # 调整标签大小以适应图像
                self.scroll_area_layout.addWidget(image_label, row, col)
                col += 1
                if col >= 3:  # 每行显示3张图像
                    col = 0
                    row += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
