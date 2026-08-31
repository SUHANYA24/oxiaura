# Front-end prompt — Product catalog + proposal product picker

Paste everything below the line into your front-end AI tool (v0, Cursor, Claude, Lovable, …).
It is written against the **live contract** of this Flask backend (Phase 13), so every field
name, enum value, query parameter and status code below is what the API actually returns.

Target stack: **React 18 + Tailwind CSS**.

---

## Context

You are building the **Product catalog** screens for PlantVest, an internal plantation-investment
management system. The backend is a Flask REST API at `http://localhost:5000/api/v1`. It is
already built and cannot be changed — match its contract exactly.

A **product** is an investment offering (e.g. "Teak Plantation Unit": min LKR 50,000, 180 months,
12.5% p.a.). The catalog is **prebuilt reference data managed by admins**. Sales reps and head-office
staff can only read it — they select a product when submitting a customer proposal.

Build these three things:

1. **Products list page** — search, filters, pagination, admin-only action buttons.
2. **Product create/edit modal** — one form serving both, plus deactivate and retire confirmations.
3. **Product picker** — a `<select>`-style field inside the existing proposal form, plus the
   "revise proposal" flow.

---

## Authentication & roles

Auth is **JWT bearer**. `POST /auth/login` with `{ "email", "password" }` returns:

```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "user": {
    "id": 1,
    "full_name": "System Administrator",
    "email": "admin@plantvest.local",
    "role": "admin",
    "branch_id": 1,
    "is_active": true,
    "created_at": "2026-08-31T10:00:00"
  }
}
```

Send `Authorization: Bearer <access_token>` on every request below. Read the role from
`user.role` — one of `"admin"`, `"head_office_staff"`, `"sales_rep"`.

**The access rule that drives the whole UI:**

| Action | admin | head_office_staff | sales_rep |
|---|---|---|---|
| `GET /products` (list) | ✅ | ✅ | ✅ |
| `GET /products/{id}` (detail) | ✅ | ✅ | ✅ |
| `POST /products` (create) | ✅ | ❌ 403 | ❌ 403 |
| `PUT /products/{id}` (edit / toggle active) | ✅ | ❌ 403 | ❌ 403 |
| `DELETE /products/{id}` (retire) | ✅ | ❌ 403 | ❌ 403 |

Non-admins must **not see** the "New product", "Edit", "Deactivate" or "Retire" controls at all —
hide them, don't just disable them. The API enforces this independently, so treat a `403` as a
bug-or-stale-session case and show a toast, never a crash.

---

## Endpoint contract

### `GET /api/v1/products` — list (all roles)

Query parameters, all optional:

| Param | Type | Notes |
|---|---|---|
| `page` | int | default `1`. Invalid values silently fall back to the default. |
| `per_page` | int | default `20`, **server-capped at 100** — echo back `pagination.per_page`, don't assume yours was honoured. |
| `category` | string | one of `teak` `agarwood` `coconut` `mixed` `other`. Anything else → **422**. |
| `is_active` | bool | accepts `true/1/yes` or `false/0/no`. Unrecognized values are ignored (no filter). |
| `search` | string | case-insensitive partial match over `name`, `product_code` and `description`. |

Retired (soft-deleted) products are **never** returned. Results are newest-first.

```json
{
  "items": [
    {
      "id": 1,
      "product_code": "PRD-1001",
      "name": "Teak Plantation Unit",
      "category": "teak",
      "description": "15-year teak growth unit with annual yield reporting.",
      "min_investment": "50000.00",
      "max_investment": "500000.00",
      "duration_months": 180,
      "interest_rate": 12.5,
      "is_active": true,
      "created_at": "2026-08-31T10:00:00",
      "updated_at": "2026-08-31T10:00:00"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 3,
    "pages": 1,
    "has_next": false,
    "has_prev": false
  }
}
```

**Field notes that matter for rendering:**

- `min_investment` / `max_investment` are **decimal strings**, not numbers — `"50000.00"`.
  Never do arithmetic on them with `+`; parse explicitly when you need to compare.
- `max_investment` is **nullable** — render "No upper limit" when it is `null`.
- `interest_rate` is a **number** and means **percent per annum** — render `12.5%`.
- `duration_months` is months — render "180 months (15 yrs)".
- `product_code` is server-generated (`PRD-1001`, `PRD-1002`, …). Never editable, never sent.

### `GET /api/v1/products/{id}` — detail (all roles)

Same fields as a list item **plus**:

```json
{ "...": "...", "proposal_count": 4 }
```

`proposal_count` is how many proposals reference this product. Use it in the retire confirmation
("4 proposals reference this product") — it is the reason retiring is non-destructive.

Missing or retired id → **404**.

### `POST /api/v1/products` — create (**admin only**) → `201`

Send **exactly** these keys. Any extra key → **422** (the API rejects unknown fields):

```json
{
  "name": "Agarwood Growth Unit",
  "category": "agarwood",
  "description": "20-year agarwood unit, higher return and longer lock-in.",
  "min_investment": "100000.00",
  "max_investment": "1000000.00",
  "duration_months": 240,
  "interest_rate": 14.0,
  "is_active": true
}
```

- **Required:** `name`, `category`, `min_investment`, `duration_months`, `interest_rate`.
- **Optional:** `description` (nullable), `max_investment` (nullable — **omit the key entirely**
  for "no ceiling"; don't send `""`), `is_active` (defaults to `true`).
- **Never send:** `product_code`, `id`, `is_deleted`, `created_at`, `updated_at` → **422**.
- `name` is trimmed server-side and must be **unique case-insensitively** → **409** otherwise.

Returns the created product (list-item shape, no `proposal_count`).

### `PUT /api/v1/products/{id}` — update (**admin only**) → `200`

**Partial update** — send only changed keys. Same field rules as create; nothing is required.
This is also the endpoint for the active toggle:

```json
{ "is_active": false }
```

Returns the updated product. `409` on renaming onto another product's name; renaming a product
to its own current name is fine (`200`).

### `DELETE /api/v1/products/{id}` — retire (**admin only**) → `200`

```json
{ "message": "Product deleted." }
```

This is a **soft delete**. The row survives so existing proposals keep resolving; the product just
vanishes from every read. Deleting twice → **404**. Retiring **frees its name for reuse**.

---

## Error envelope

Two shapes. Handle both in one helper.

**Schema validation (422)** — field-keyed, from Marshmallow:

```json
{
  "error": "validation_error",
  "messages": {
    "min_investment": ["Must be greater than 0."],
    "category": ["Must be one of: teak, agarwood, coconut, mixed, other."],
    "hacker": ["Unknown field."]
  }
}
```

Map each key onto the matching input's inline error. Note `messages` values are **arrays**.

**Everything else** — single message:

```json
{ "error": "conflict", "message": "A product named 'Teak Plantation Unit' already exists." }
```

| Status | `error` | Where it comes from | UI response |
|---|---|---|---|
| 401 | `authorization_required` / `invalid_token` / `token_expired` / `token_revoked` | missing or dead token | refresh via `POST /auth/refresh`, else redirect to login |
| 403 | `forbidden` | non-admin attempted a write | toast "You don't have permission to manage products." |
| 404 | `not_found` | unknown or retired product | toast + drop the row from the list |
| 409 | `conflict` | duplicate product name | **inline error on the `name` field**, not a toast |
| 422 | `validation_error` | bad field, unknown field, or business rule | inline per-field errors (see below) |

**Important — the amount-window rule reaches you in *both* shapes**, depending on where it was
caught. When your request contains **both** amounts, the schema catches it and you get the
field-keyed shape:

```json
{
  "error": "validation_error",
  "messages": { "max_investment": ["max_investment must be greater than or equal to min_investment."] }
}
```

When a **partial** update contains only one of them, the schema can't see the conflict, so the
service checks the *merged* window and raises the single-message shape:

```json
{ "error": "validation_error", "message": "max_investment must be greater than or equal to min_investment." }
```

So `PUT {"max_investment": "100.00"}` against a product whose stored `min_investment` is
`50000.00` is a 422 with no `messages` key at all. Your normalizer must route a message-only 422
somewhere visible — show it near the amount fields, not as an anonymous toast.

---

## Screen 1 — Products list

Route `/products`. Layout: page header, filter bar, table, pagination footer.

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Products                                            [ + New product ]     │  ← button: admin only
│  Investment offerings available to sales reps.                             │
├────────────────────────────────────────────────────────────────────────────┤
│  [🔍 Search name or code…]  [Category ▾]  [Status ▾]         3 products    │
├──────────┬──────────────────────┬──────────┬─────────────┬────────┬────────┤
│ CODE     │ PRODUCT              │ CATEGORY │ INVESTMENT  │ TERM   │ RATE   │
├──────────┼──────────────────────┼──────────┼─────────────┼────────┼────────┤
│ PRD-1001 │ Teak Plantation Unit │ ● Teak   │ 50K – 500K  │ 180 mo │ 12.50% │
│          │ 15-year teak grow…   │          │             │ 15 yrs │        │
│          │                      │          │        [ Active ]  [ ⋯ ]      │
├──────────┼──────────────────────┼──────────┼─────────────┼────────┼────────┤
│ PRD-1003 │ Coconut Estate Share │ ● Coconut│ 25K – no cap│ 120 mo │  9.75% │
│          │                      │          │             │ 10 yrs │        │
│          │                      │          │      [ Inactive ]  [ ⋯ ]      │
└──────────┴──────────────────────┴──────────┴─────────────┴────────┴────────┘
   Showing 1–3 of 3                                        ‹ Prev   1   Next ›
```

Requirements:

- **Search** — debounce 300 ms, then refetch with `?search=`. Reset `page` to 1 on any filter change.
- **Category filter** — "All categories" + the five enum values. Send the raw lowercase value.
  Show a coloured dot per category (teak amber, agarwood violet, coconut emerald, mixed sky,
  other slate).
- **Status filter** — "All" / "Active" / "Inactive" → omit / `is_active=true` / `is_active=false`.
- **Amounts** — format the decimal strings as `LKR 50,000` (compact `50K`/`1.5M` in the table is
  fine); when `max_investment` is `null` render `25,000 +` or "no cap".
- **`is_active` badge** — emerald "Active" / slate "Inactive". Inactive products stay listed
  (admins need to find them to switch them back on) but render the row at `opacity-60`.
- **Row actions (`⋯` menu) — admin only:** Edit, Deactivate/Activate, Retire.
- **Pagination** — drive Prev/Next off `pagination.has_prev` / `has_next`, and page count off
  `pagination.pages`. Show `Showing X–Y of pagination.total`.
- **Empty states** — distinguish "no products yet" (offer + New product to admins) from
  "no products match these filters" (offer a Clear filters button).
- **Loading** — skeleton rows on first load; keep the old rows visible with a subtle overlay
  while refetching after a filter change, so the table doesn't flash.
- Make the row clickable → product detail drawer/page, which calls `GET /products/{id}` and shows
  `description` in full plus `proposal_count`.

## Screen 2 — Create / edit modal

One `<ProductFormModal mode="create" | "edit" product={…} />`. Admin only.

Fields, in order:

1. **Name** — text, required, max 150. Trim before sending.
2. **Category** — select, required, the five enum values (label them Title Case, send lowercase).
3. **Description** — textarea, optional, 3 rows.
4. **Minimum investment** — number input with an `LKR` prefix, required, must be **> 0**.
5. **Maximum investment** — number input, optional, with a "No upper limit" checkbox. When checked,
   disable the input and **omit the key** from the payload.
6. **Duration (months)** — number, required, integer **≥ 1**. Show a live "= 15 years" hint.
7. **Interest rate (% p.a.)** — number, required, **≥ 0**, one decimal step.
8. **Active** — toggle, defaults on. Helper text: *"Inactive products stay in the catalog but
   cannot be selected on new proposals."*

Behaviour:

- **Client-side validation mirrors the server**, so the common cases never round-trip: required
  fields, `min > 0`, `duration ≥ 1`, `rate ≥ 0`, and `max ≥ min` when both are present. Still handle
  the server's 409/422 — the client cannot know about name collisions.
- **Send amounts as strings** (`String(value)`) to match the API's decimal-string fields, and keep
  `duration_months` / `interest_rate` as **numbers**.
- **In edit mode, send only dirty fields.** Diff against the loaded product. This is both what the
  API expects (partial `PUT`) and what avoids re-tripping the amount-window rule on untouched fields.
- `product_code` is displayed read-only in edit mode (grey, top-right of the modal). Never in the payload.
- On 409, focus the name input and show the server's message (*"A product named 'X' already
  exists."*) under it.
- On success: close, toast (`Product created` / `Product updated`), refetch the list.
- Disable the submit button while in flight; never let a double-click create two products.

**Deactivate / Activate** — a small confirm dialog, not the full modal. It sends
`PUT /products/{id}` with `{ "is_active": false }` (or `true`). Copy:

> **Deactivate "Teak Plantation Unit"?**
> Sales reps will no longer be able to select it on new proposals. Existing proposals are unaffected.
> You can reactivate it at any time.

**Retire (delete)** — destructive-styled confirm dialog, sends `DELETE /products/{id}`. Fetch
`GET /products/{id}` first so you can name the count. Copy:

> **Retire "Teak Plantation Unit"?**
> It will be removed from the catalog and can no longer be selected. **4 existing proposals** that
> reference it are kept intact and will still display correctly.
> This does not delete historical data, but it cannot be undone from this screen.

Require the admin to click a red **Retire product** button; don't make Retire the default focus.

## Screen 3 — Product picker on the proposal form

Submitting a proposal now **requires** a product. `POST /api/v1/proposals`:

```json
{
  "customer_id": 12,
  "product_id": 1,
  "proposed_amount": "75000.00",
  "notes": "Client prefers quarterly payouts."
}
```

- `product_id` is **required** → omitting it returns `422` with `messages.product_id`.
- **Do not send `product_type`.** It used to be a free-text field; it is now server-owned and
  sending it returns **422** ("Unknown field"). The response carries it back as a *snapshot* of
  the product's name at submission time.

Response (`201`):

```json
{
  "id": 7,
  "customer_id": 12,
  "sales_rep_id": 3,
  "product_id": 1,
  "proposed_amount": "75000.00",
  "product_type": "Teak Plantation Unit",
  "workflow_status": "submitted",
  "notes": "Client prefers quarterly payouts.",
  "submitted_at": "2026-08-31T11:00:00"
}
```

Picker requirements:

- Load options from `GET /products?is_active=true&per_page=100`. **Always filter to active** —
  selecting an inactive product is rejected with `422` and the message
  *"Product 'X' is not currently available for new proposals."*
- Render each option as two lines: **name** + `product_code`, then the terms
  (`LKR 50,000 – 500,000 · 180 months · 12.5% p.a.`). A searchable combobox beats a bare `<select>`
  once there are more than ~10 products.
- On selection, show a **terms summary card** under the field, and use the product's window to
  give the amount input a soft client-side hint: if `proposed_amount` falls outside
  `min_investment … max_investment`, show an amber warning ("Below this product's 50,000 minimum")
  — **a warning, not a block.** The backend does not currently enforce the proposal amount against
  the product window, so do not invent a hard error the API won't produce.
- Handle the empty catalog: if no active products come back, disable the submit button and show
  *"No products are currently available. Ask an administrator to add one."*

### Proposal detail — snapshot vs. current

`GET /api/v1/proposals/{id}` returns the proposal plus a nested product:

```json
{
  "id": 7,
  "product_id": 1,
  "product_type": "Teak Plantation Unit",
  "workflow_status": "rep_review",
  "product": {
    "id": 1,
    "product_code": "PRD-1001",
    "name": "Teak Unit (2026 terms)",
    "category": "teak",
    "duration_months": 180,
    "interest_rate": 12.5,
    "is_active": true
  },
  "customer": { "id": 12, "customer_code": "C-1001", "full_name": "Nimal Perera" }
}
```

These two fields **can legitimately disagree**, and that is the point:

- `product_type` = the name **as proposed** (a frozen snapshot).
- `product.name` = the product's **current** name, after any admin edit.

Render `product_type` as the proposal's product, and when it differs from `product.name`, add a
small muted note: *"Product since renamed to 'Teak Unit (2026 terms)'."* Never silently replace the
snapshot with the current name — the snapshot is the historical record.

`product` is nullable (pre-catalog proposals), so guard every access.

### Revise proposal — `PUT /api/v1/proposals/{id}`

Editable **only while `workflow_status` is `submitted` or `rep_review`.** Partial body, only these
three keys (anything else, including `customer_id`, → **422**):

```json
{ "product_id": 2, "proposed_amount": "90000.00", "notes": "Client upgraded." }
```

- Changing `product_id` **re-snapshots** `product_type` to the new product's name — reflect the
  updated value from the response rather than guessing.
- Once the proposal reaches `ho_review`, `approved` or `rejected` the API returns **422**:
  *"…can no longer be edited. Editable stages: submitted, rep_review."*
  So **hide the Edit control entirely** for those statuses instead of letting the user discover it.
- A rep may only revise their own proposals → otherwise **403**. Admins may revise any.

---

## Implementation notes

- **File layout**

  ```
  src/
    api/client.js            # fetch wrapper: base URL, bearer token, refresh-on-401, error normalizer
    api/products.js          # listProducts, getProduct, createProduct, updateProduct, deleteProduct
    hooks/useProducts.js     # list + filter state (query params are the source of truth)
    hooks/useAuth.js         # current user + role helpers: const { isAdmin } = useAuth()
    pages/ProductsPage.jsx
    components/products/ProductsTable.jsx
    components/products/ProductFilters.jsx
    components/products/ProductFormModal.jsx
    components/products/ConfirmDialog.jsx
    components/products/CategoryBadge.jsx
    components/proposals/ProductPicker.jsx
    utils/money.js           # decimal-string → display, and back
  ```

- **One error normalizer, used everywhere.** Turn both response shapes into
  `{ status, code, message, fieldErrors }` where `fieldErrors` is `{ field: "first message" }`.
  Every form then reads `fieldErrors` the same way regardless of which shape the server sent.
- **Query params are the source of truth** for list state (search/category/status/page), so a
  filtered view is linkable and survives a refresh.
- **Gate on role, once.** `const { isAdmin } = useAuth()` — use it for every write control rather
  than re-deriving the check in each component.
- **Tailwind only** — no component library. Use `rounded-lg border border-slate-200`,
  `text-slate-500` for secondary text, `bg-slate-900 text-white` for primary buttons,
  `bg-red-600` for the retire action. Support dark mode via `dark:` variants.
- **Accessibility** — modals: focus trap, `Esc` to close, `aria-modal="true"`, focus returns to the
  trigger. Inputs: real `<label>` elements, `aria-invalid` + `aria-describedby` on errors.
  The `⋯` menu must be keyboard-navigable.
- Responsive: the table collapses to stacked cards below `md`.

## Definition of done

- A `sales_rep` sees the products list with **no** create/edit/retire controls anywhere.
- An `admin` can create a product; its `product_code` appears as `PRD-100n` without ever being typed.
- Submitting a duplicate name shows an inline error on the name field, not a toast.
- Setting max below min shows the server's window message beside the amount fields — including on a
  partial edit that only touches one of the two.
- Retiring a product removes it from the list, and a proposal that referenced it still renders its
  product correctly.
- A proposal cannot be submitted without a product; inactive products never appear in the picker.
- A renamed product leaves an already-submitted proposal showing its original `product_type`, with
  the "since renamed" note.
- The Edit control is absent on any proposal at `ho_review` or later.
