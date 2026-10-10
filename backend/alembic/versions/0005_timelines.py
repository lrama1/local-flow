"""timelines

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-10 11:50:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

import localflow.models.base
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "timelines",
        sa.Column("created_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", localflow.models.base.UTCDateTime(), nullable=False),
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(length=26), nullable=False),
        sa.Column("project_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("tracks_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("timelines", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_timelines_project_id"), ["project_id"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("timelines", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_timelines_project_id"))
    op.drop_table("timelines")
