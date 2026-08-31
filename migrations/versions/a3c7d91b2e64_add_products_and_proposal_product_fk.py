"""add products catalog and proposals.product_id FK

Revision ID: a3c7d91b2e64
Revises: f325d629da73
Create Date: 2026-08-31 22:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a3c7d91b2e64'
down_revision = 'f325d629da73'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'products',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_code', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=150), nullable=False),
        sa.Column(
            'category',
            sa.Enum(
                'teak', 'agarwood', 'coconut', 'mixed', 'other',
                name='productcategory',
            ),
            nullable=False,
        ),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('min_investment', sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column('max_investment', sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column('duration_months', sa.Integer(), nullable=False),
        sa.Column('interest_rate', sa.Float(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('product_code', name=op.f('uq_products_product_code')),
    )
    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_products_name'), ['name'], unique=False)
        batch_op.create_index(batch_op.f('ix_products_is_active'), ['is_active'], unique=False)
        batch_op.create_index(batch_op.f('ix_products_is_deleted'), ['is_deleted'], unique=False)

    # Nullable, so the proposal rows that predate the catalog keep loading on
    # their product_type text alone. New proposals are required to supply a
    # product_id at the schema layer (ProposalCreateSchema), not here.
    # The FK is named explicitly because SQLite batch mode cannot create an
    # unnamed constraint.
    with op.batch_alter_table('proposals', schema=None) as batch_op:
        batch_op.add_column(sa.Column('product_id', sa.Integer(), nullable=True))
        batch_op.create_index(
            batch_op.f('ix_proposals_product_id'), ['product_id'], unique=False
        )
        batch_op.create_foreign_key(
            'fk_proposals_product_id_products', 'products', ['product_id'], ['id']
        )


def downgrade():
    with op.batch_alter_table('proposals', schema=None) as batch_op:
        batch_op.drop_constraint('fk_proposals_product_id_products', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_proposals_product_id'))
        batch_op.drop_column('product_id')

    with op.batch_alter_table('products', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_products_is_deleted'))
        batch_op.drop_index(batch_op.f('ix_products_is_active'))
        batch_op.drop_index(batch_op.f('ix_products_name'))

    op.drop_table('products')
