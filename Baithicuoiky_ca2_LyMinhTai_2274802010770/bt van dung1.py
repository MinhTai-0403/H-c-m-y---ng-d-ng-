import datetime
class datetime:
    def __init__(self,ngay_gio):
        self.ngay_gio=datetime.datetime.now()

    def tinh_ngay_gio(self):
        return self.ngay_gio
        print("ngày và giờ hiện tại: {self.ngay_gio}")
        