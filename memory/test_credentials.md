# Test Credentials

This project does NOT use authentication. Predictions are stored anonymously per-device using a `localStorage` session id (key `bharat_estate_sid`, generated on first load). For testing, you can use any string as `session_id` query param on the `/api/history` endpoints.

Example session_id used during dev: `sid_test_001`
