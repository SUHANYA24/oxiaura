"""Service layer.

Routes validate input (schemas) and delegate business logic here; services own
database access via the ORM and never build raw SQL strings (BUILD_SPEC
section 3, rule 6).
"""
