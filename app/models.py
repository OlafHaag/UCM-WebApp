"""
Models for database tables, columns and relationships.

When using ForeignKey:
    The target column must have a unique constraint in order to build a relationship between two tables.
"""

from datetime import datetime

from app.extensions import db


class Device(db.Model):
    """ Information about a device regarding display size and platform. """
    __tablename__ = 'devices'
    __table_args__ = {'comment': "This table holds in each row information about an individual device."}
    
    id = db.Column(db.String(8), primary_key=True, comment="Unique identifier for a device.")
    screen_x = db.Column(db.Integer, unique=False, nullable=True,
                         comment="Resolution of the horizontal axis of the screen")
    screen_y = db.Column(db.Integer, unique=False, nullable=True,
                         comment="Resolution of the vertical axis of the screen")
    dpi = db.Column(db.Float, unique=False, nullable=True, comment="Pixel density of the display per inch.")
    density = db.Column(db.Float, unique=False, nullable=True,
                        comment="Density of the screen. This value is 1 by default on desktops but varies on android "
                                "depending on the screen.")
    aspect_ratio = db.Column(db.Float, unique=False, nullable=True, comment="Aspect ratio of the screen.")
    size_x = db.Column(db.Float, unique=False, nullable=True,
                       comment="Estimated horizontal size of the screen in centimeters.")
    size_y = db.Column(db.Float, unique=False, nullable=True,
                       comment="Estimated vertical size of the screen in centimeters.")
    platform = db.Column(db.String(16), unique=False, nullable=True,
                         comment="Abbreviated identifier for operating system running on the device.")
    
    users = db.relationship('User', backref='device')
    
    def __repr__(self):
        return f"Device(id='{self.id}', screen_x={self.screen_x}, screen_y={self.screen_y}, dpi={self.dpi}, " \
            f"density={self.density}, aspect_ratio={self.aspect_ratio}, size_x={self.size_x}, size_y={self.size_y}, " \
            f"platform='{self.platform}')"


class User(db.Model):
    """ Theoretically a user could use multiple devices (many-to-many), but it's ignored here. """
    __tablename__ = 'users'
    __table_args__ = {'comment': "This table holds in each row information about a user."}
    
    id = db.Column(db.String(32), primary_key=True, comment="Unique identifier for a user.")
    device_id = db.Column(db.String(8), db.ForeignKey('devices.id'), nullable=False, comment="Device used by user.")
    blocks_ct = db.relationship('CircleTaskBlock', backref='user')
    trials_ct = db.relationship('CircleTaskTrial', backref='user')  # Shortcut to trials. ToDo: Really necessary?
    
    def __repr__(self):
        return f"User(id='{self.id}', device_id='{self.device_id}')"


class CircleTaskBlock(db.Model):
    """ This represents meta information about 1 block in the circle task. """
    __tablename__ = 'circletask_blocks'
    __table_args__ = {'comment': "This table holds in each row information about a block in the Circle Task study. "
                                 "Blocks can be associated with one another as belonging to the same session. "
                                 "A session is a single run through the study with all its consecutive blocks. "
                                 "Each block is comprised of trials."
                      }
    
    # The combination of user_id and hash should* be unique and could be used as a composite key (and drop id).
    # time column can't be used, as it may not be unique, even though it's unlikely.
    # *It's highly unlikely that someone produces the exact same dataset, but not impossible.
    id = db.Column(db.Integer, primary_key=True, comment="Unique identifier for a block.")
    user_id = db.Column(db.String(32), db.ForeignKey('users.id'), nullable=False,
                        comment="User who performed the task.")
    session_uid = db.Column(db.String(32), unique=False, nullable=False,
                            comment="The session a block belongs to. Sessions consist of consecutive blocks in a "
                                    "single run through the study.")
    nth_session = db.Column(db.Integer, unique=False, nullable=False, default=1,
                            comment="Chronological order of a session per user. How many times did a user upload data?")
    nth_block = db.Column(db.Integer, unique=False, nullable=True, default=1,
                          comment="Chronological order of a block within a session.")
    treatment = db.Column(db.String(120), unique=False, nullable=True,
                         comment="Which degree of freedom had a constraint on it during a block. Independent variable.")
    warm_up = db.Column(db.Float, unique=False, nullable=False, default=0.5,
                        comment="Time before each trial to prepare, in seconds.")
    trial_duration = db.Column(db.Float, unique=False, nullable=False, default=2.0,
                               comment="Time given for performing the task for each trial, in seconds.")
    cool_down = db.Column(db.Float, unique=False, nullable=False, default=0.5,
                          comment="Time after each trial to give feedback, in seconds.")
    time = db.Column(db.Float, unique=False, nullable=True,
                     comment="Time at which a block was finished, in seconds since epoch.")
    time_iso = db.Column(db.DateTime, unique=False, nullable=False, default=datetime.utcnow,
                         comment="Time at which a block was finished, in ISO format.")
    hash = db.Column(db.String(32), unique=True, nullable=False,
                     comment="MD5 hash value for all the trials of a block. Used to check integrity of submitted data.")
    
    trials = db.relationship('CircleTaskTrial', backref='block')

    def __repr__(self):
        return f"CircleTaskBlock(user_id='{self.user_id}', session_uid={self.session_uid}, " \
               f"nth_session={self.nth_session}, nth_block={self.nth_block}, treatment='{self.treatment}', "\
               f"time={self.time}, time_iso='{self.time_iso}', hash='{self.hash}')"


class CircleTaskTrial(db.Model):
    """ Variables specific to circle task. """
    __tablename__ = 'circletask_trials'
    __table_args__ = {'comment': "This table holds in each row information about the outcome of a single trial from "
                                 "the Circle Task study. Trials belong to a block, which are part of a complete run "
                                 "through the study."}
    
    # Can't use ForeignKeyConstraint as composite key,
    # because user_id, session and block are not unique keys in reference table.
    id = db.Column(db.Integer, primary_key=True, comment="Unique identifier of a trial.")
    # ToDo: user_id rly necessary? -> self.block_id.user_id or get-method
    user_id = db.Column(db.String(32), db.ForeignKey('users.id'), nullable=False,
                        comment="User who performed the task.")
    block_id = db.Column(db.Integer, db.ForeignKey('circletask_blocks.id'), nullable=False,
                         comment="The block a trial belongs to.")
    # Since nth_block is a non-unique property in CircleTaskBlock we cannot place its value here,
    # but have to go through block_id.
    trial = db.Column(db.Integer, unique=False, nullable=False, comment="Chronological order of trial within a block.")
    df1 = db.Column(db.Float, unique=False, nullable=True,
                    comment="Value of degree of freedom 1 at end of trial. Dependent variable.")
    df2 = db.Column(db.Float, unique=False, nullable=True,
                    comment="Value of degree of freedom 2 at end of trial. Dependent variable.")
    df1_grab = db.Column(db.Float, unique=False, nullable=True,
                         comment="Delta time in seconds after trial onset at which df1 slider was grabbed.")
    df1_release = db.Column(db.Float, unique=False, nullable=True,
                            comment="Delta time in seconds after trial onset at which df1 slider was released, either "
                                    "by the user or by the end of the countdown.")
    df1_duration = db.Column(db.Float, unique=False, nullable=True, comment="Duration of the df1 grab, in seconds.")
    df2_grab = db.Column(db.Float, unique=False, nullable=True,
                         comment="Delta time in seconds after trial onset at which df2 slider was grabbed.")
    df2_release = db.Column(db.Float, unique=False, nullable=True,
                            comment="Delta time in seconds after trial onset at which df2 slider was released, either "
                                    "by the user or by the end of the countdown.")
    df2_duration = db.Column(db.Float, unique=False, nullable=True, comment="Duration of the df2 grab, in seconds.")
    sum = db.Column(db.Float, unique=False, nullable=True,
                    comment="Sum of values for df1 and df2 at end of trial. Dependent variable.")

    def __init__(self, **kwargs):
        super(CircleTaskTrial, self).__init__(**kwargs)
        self.df1_duration = kwargs['df1_release'] - kwargs['df1_grab']
        self.df2_duration = kwargs['df2_release'] - kwargs['df2_grab']
        self.sum = kwargs['df1'] + kwargs['df2']
        
    def __repr__(self):
        return f"CircleTaskTrial(user_id='{self.user_id}', block_id={self.block_id}, trial={self.trial}, " \
               f"df1={self.df1}, df1_grab={self.df1_grab}, df1_release={self.df1_release}, " \
               f"df2={self.df2}, df2_grab={self.df2_grab}, df2_release={self.df2_release})"
