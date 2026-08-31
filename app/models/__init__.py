"""SQLAlchemy models package.

Importing this package pulls in every model so that
``db.metadata`` is fully populated. Flask-Migrate / Alembic relies on this
to autogenerate migrations, so the factory imports this package during
application setup.
"""

from .agreement import Agreement, AgreementStatus
from .branch import Branch
from .customer import Customer, CustomerStatus
from .document import Document, DocType, VerificationStatus
from .employee_target import EmployeeTarget
from .fraud_log import FraudLog
from .notification import Notification
from .product import Product, ProductCategory
from .proposal import Proposal, ProposalWorkflowStatus
from .user import User, UserRole

__all__ = [
    "Agreement",
    "AgreementStatus",
    "Branch",
    "Customer",
    "CustomerStatus",
    "Document",
    "DocType",
    "VerificationStatus",
    "EmployeeTarget",
    "FraudLog",
    "Notification",
    "Product",
    "ProductCategory",
    "Proposal",
    "ProposalWorkflowStatus",
    "User",
    "UserRole",
]
