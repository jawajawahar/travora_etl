# Entrance fee sources

Collected 2026-09-27 for `config/entrance_fees.yaml`. Each page was read once;
no automated crawling. The CCF e-ticketing system
(https://eservices.ccf.gov.lk) is a booking service, not a data service, and was
not accessed programmatically.

| # | Source | Type | Coverage | Stored |
|---|---|---|---|---|
| 1 | Central Cultural Fund, "Prices of the Tickets for Archeological Sites and Museums", https://ccf.gov.lk/ticket-issuance/ (table at https://ccf.gov.lk/is/index.php) | **Official** | 16 sites and museums; foreign full and half ticket, USD and LKR, including 18% VAT | `01_ccf_ticket_prices_2026-09-27.html`, `01b_ccf_ticket_issuance_page_2026-09-27.html` |
| 2 | slvoyo.com, "Sri Lanka Entrance (Ticket) Fees for Foreign Tourists (2026 Updated Prices)", updated 2026-04-03 | Secondary | Temples, gardens, national parks; some SAARC and child rates | URL and extracted figures only |
| 3 | tuktukrental.com, "Entrance Fees Sri Lanka 2025-2026", last verified 2026-03-30 (319 LKR/USD) | Secondary | Widest list; LKR prices; national-park all-in costs | URL and extracted figures only |
| 4 | hopscotchtravels.com, "Entrance Fees for Sri Lanka's Top Attractions (2026 Guide)", 2025-09-23 | Secondary | The only source with local (Sri Lankan) prices, for 10 sites | URL and extracted figures only |

## How figures were chosen

1. The official CCF figure is used wherever CCF lists the site.
2. Otherwise the figure given by most secondary sources is used; where they
   disagree, the row is marked `conflict: true` and every figure is kept in
   its `note`, so the choice can be checked or changed.
3. USD amounts are converted at the CCF table's own rate (334 LKR per USD),
   so every row uses one rate.

## Limits to state in the dissertation

- Only CCF is an official source; the other three are travel guides and may be
  out of date. Prices follow the exchange rate and can change by gazette.
- Fees differ by visitor type (foreign, SAARC, local). Local prices were
  found for only a handful of sites.
- National-park figures are entry fees; jeep hire (about LKR 7,000-17,000) is
  extra and is not an entrance fee.
- Places not in this file keep their category default, still flagged
  `fee_estimated = true`.
