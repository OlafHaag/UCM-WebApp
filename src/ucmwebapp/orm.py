from datetime import datetime
from src.ucmwebapp import db


# ToDo: relationships, backrefs
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(8), unique=True, nullable=False)
    
    def __repr__(self):
        return f"User('{self.id}', '{self.device_id}')"


class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session = db.Column(db.Integer, unique=False, nullable=False, default=1)
    block = db.Column(db.Integer, unique=False, nullable=True, default=1)
    treatment = db.Column(db.String(120), unique=False, nullable=True)
    time = db.Column(db.Float, unique=False, nullable=True)
    time_iso = db.Column(db.DateTime, unique=False, nullable=False, default=datetime.utcnow)
    hash = db.Column(db.String(32), unique=True, nullable=False)
    
    def __repr__(self):
        return f"Session('{self.id}', '{self.session}', '{self.block}', '{self.treatment}', '{self.time}', " \
            f"'{self.time_iso}', '{self.hash}')"


class Device(db.Model):
    device_id = db.Column(db.String(8), primary_key=True)
    screen_x = db.Column(db.Integer, unique=False, nullable=True)
    screen_y = db.Column(db.Integer, unique=False, nullable=True)
    dpi = db.Column(db.Float, unique=False, nullable=True)
    aspect_ratio = db.Column(db.Float, unique=False, nullable=True)
    size_x = db.Column(db.Float, unique=False, nullable=True)
    size_y = db.Column(db.Float, unique=False, nullable=True)
    platform = db.Column(db.String(16), unique=False, nullable=True)
    
    def __repr__(self):
        return f"Device('{self.device_id}', '{self.screen_x}x{self.screen_y}', '{self.dpi}', '{self.aspect_ratio}', " \
            f"'{self.size_x}x{self.size_y}', '{self.platform}')"


class CircleTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session = db.Column(db.Integer, unique=False, nullable=False, default=1)
    block = db.Column(db.Integer, unique=False, nullable=False, default=1)
    trial = db.Column(db.Integer, unique=False, nullable=False)
    df1 = db.Column(db.Float, unique=False, nullable=True)
    df2 = db.Column(db.Float, unique=False, nullable=True)
    sum = db.Column(db.Float, unique=False, nullable=True)
    
    def __repr__(self):
        return f"CircleTask('{self.id}', '{self.session}', '{self.block}', '{self.trial}', " \
            f"'{self.df1}', '{self.df2}','{self.sum}')"
