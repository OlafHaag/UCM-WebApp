"""'density'

Revision ID: 26593a115a73
Revises: 36dd0a57afb9
Create Date: 2019-07-20 12:41:29.747301

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '26593a115a73'
down_revision = '36dd0a57afb9'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('devices', sa.Column('density', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('devices', 'density')
    # ### end Alembic commands ###
