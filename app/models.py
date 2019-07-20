"""
Models for database tables, columns and relationships.

When using ForeignKey:
    The target column must have a unique constraint in order to build a relationship between two tables.
"""

from datetime import datetime

from app.extensions import db


class Device(db.Model):
    __tablename__ = 'devices'
    
    id = db.Column(db.String(8), primary_key=True)
    screen_x = db.Column(db.Integer, unique=False, nullable=True)
    screen_y = db.Column(db.Integer, unique=False, nullable=True)
    dpi = db.Column(db.Float, unique=False, nullable=True)
    density = db.Column(db.Float, unique=False, nullable=True)
    aspect_ratio = db.Column(db.Float, unique=False, nullable=True)
    size_x = db.Column(db.Float, unique=False, nullable=True)
    size_y = db.Column(db.Float, unique=False, nullable=True)
    platform = db.Column(db.String(16), unique=False, nullable=True)
    
    users = db.relationship('User', backref='device')
    
    def __repr__(self):
        return f"Device(id='{self.id}', screen_x={self.screen_x}, screen_y={self.screen_y}, dpi={self.dpi}, " \
            f"density={self.density}, aspect_ratio={self.aspect_ratio}, size_x={self.size_x}, size_y={self.size_y}, " \
            f"platform='{self.platform}')"


class User(db.Model):
    """ Theoretically a user could use multiple devices, but it's ignored here. """
    __tablename__ = 'users'
    
    id = db.Column(db.String(32), primary_key=True)
    device_id = db.Column(db.String(8), db.ForeignKey('devices.id'), nullable=False)
    ct_sessions = db.relationship('CTSession', backref='user')
    trials_CT = db.relationship('CircleTask', backref='user')  # Shortcut to trials. ToDo: Really necessary?
    
    def __repr__(self):
        return f"User(id='{self.id}', device_id='{self.device_id}')"


class CTSession(db.Model):
    __tablename__ = 'ct_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(32), db.ForeignKey('users.id'), nullable=False)
    session = db.Column(db.Integer, unique=False, nullable=False, default=1)  # For chronological ordering.
    block = db.Column(db.Integer, unique=False, nullable=True, default=1)
    treatment = db.Column(db.String(120), unique=False, nullable=True)
    time = db.Column(db.Float, unique=False, nullable=True)
    time_iso = db.Column(db.DateTime, unique=False, nullable=False, default=datetime.utcnow)
    hash = db.Column(db.String(32), unique=True, nullable=False)
    
    trials_CT = db.relationship('CircleTask', backref='session_obj')

    def __repr__(self):
        return f"CTSession(user_id='{self.user_id}', session={self.session}, block={self.block}, " \
            f"treatment='{self.treatment}', time={self.time}, time_iso='{self.time_iso}', hash='{self.hash}')"


class CircleTask(db.Model):
    """ Variables specific to circle task. """
    __tablename__ = 'circle_tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(32), db.ForeignKey('users.id'), nullable=False)  # ToDo: user_id rly necessary? -> self.session.user_id
    session = db.Column(db.Integer, db.ForeignKey('ct_sessions.id'), nullable=False)
    # Since block is a non-unique property in CTSession we cannot place its value here, but have to go through session.
    trial = db.Column(db.Integer, unique=False, nullable=False)
    df1 = db.Column(db.Float, unique=False, nullable=True)
    df2 = db.Column(db.Float, unique=False, nullable=True)
    sum = db.Column(db.Float, unique=False, nullable=True)

    def __init__(self, **kwargs):
        super(CircleTask, self).__init__(**kwargs)
        self.sum = kwargs['df1'] + kwargs['df2']
        
    def __repr__(self):
        return f"CircleTask(user_id='{self.user_id}', session={self.session}, trial={self.trial}, " \
            f"df1={self.df1}, df2={self.df2})"
