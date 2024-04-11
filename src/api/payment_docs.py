PAYMENT_DOCS = """API documentation:
Endpoint: http://localhost:8000/pay/

This API is for making payments. Only GET method is allowed.

Query parameters table:
item    string  Name of the item     required

Response schema (JSON object):
results     array[object]   List of dicts with responses

Each object in the "results" key has the following schema:
service_name    string  optional
lic_acc     string     optional
route_name  string     optional
route_id    integer    optional
ranking     integer    optional
"""