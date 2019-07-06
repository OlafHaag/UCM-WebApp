from datetime import datetime
from src.ucmwebapp import db


class Device(db.Model):
    device_id = db.Column(db.String(8), primary_key=True)
    screen_x = db.Column(db.Integer, unique=False, nullable=True)
    screen_y = db.Column(db.Integer, unique=False, nullable=True)
    dpi = db.Column(db.Float, unique=False, nullable=True)
    aspect_ratio = db.Column(db.Float, unique=False, nullable=True)
    size_x = db.Column(db.Float, unique=False, nullable=True)
    size_y = db.Column(db.Float, unique=False, nullable=True)
    platform = db.Column(db.String(16), unique=False, nullable=True)
    
    users = db.relationship('User', backref='device')
    
    def __repr__(self):
        return f"Device('{self.device_id}', '{self.screen_x}x{self.screen_y}', '{self.dpi}', '{self.aspect_ratio}', " \
            f"'{self.size_x}x{self.size_y}', '{self.platform}')"


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(8), db.ForeignKey('device.id'))
    ct_sessions = db.relationship('CTSession', backref='user')
    trials_CT = db.relationship('CircleTask', backref='user')  # Shortcut to trials. ToDo: Really necessary?
    
    def __repr__(self):
        return f"User('{self.id}', '{self.device_id}')"


class CTSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(8), db.ForeignKey('user.id'))
    session = db.Column(db.Integer, unique=False, nullable=False, default=1)  # For chronological ordering.
    block = db.Column(db.Integer, unique=False, nullable=True, default=1)
    treatment = db.Column(db.String(120), unique=False, nullable=True)
    time = db.Column(db.Float, unique=False, nullable=True)
    time_iso = db.Column(db.DateTime, unique=False, nullable=False, default=datetime.utcnow)
    hash = db.Column(db.String(32), unique=True, nullable=False)
    
    trials_CT = db.relationship('CircleTask', backref='session_obj')

    def __repr__(self):
        return f"Session('{self.user_id}', '{self.session}', '{self.block}', '{self.treatment}', '{self.time}', " \
            f"'{self.time_iso}', '{self.hash}')"


class CircleTask(db.Model):
    """ Variables specific to circle task. """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(8), db.ForeignKey('user.id'))  # ToDo: user shortcut necessary?
    session = db.Column(db.Integer, db.ForeignKey('ctsession.session'))
    block = db.Column(db.Integer, db.ForeignKey('ctsession.block'))
    
    trial = db.Column(db.Integer, unique=False, nullable=False)
    df1 = db.Column(db.Float, unique=False, nullable=True)
    df2 = db.Column(db.Float, unique=False, nullable=True)
    sum = db.Column(db.Float, unique=False, nullable=True)

    def __repr__(self):
        return f"CircleTask('{self.user_id}', '{self.session}', '{self.block}', '{self.trial}', " \
            f"'{self.df1}', '{self.df2}','{self.sum}')"
