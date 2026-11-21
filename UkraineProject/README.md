# Ukraine Humanitarian Fund (UHF): projects, subgrants and localization, 2019–2026

Publisher: Civic Intelligence (https://civicintelligence.digital/)
Source extract: OCHA CBPF Data Hub, ProjectSummary API, pooled fund UKR81 (Ukraine), downloaded 27.09.2026. 2026 data are partial (allocations up to late September 2026).
Version: 1.0

## What this is
A cleaned, project-level dataset of all 465 projects funded by the Ukraine Humanitarian Fund (the OCHA-managed country-based pooled fund) from 2019 to 2026, the 747 subgrants these projects pass to subpartners, and a yearly localization indicator: how much of the fund's money reaches national NGOs directly and including subgrants.

## Files
- `uhf_projects_2019_2026.csv`: one row per project. Code, allocation year and type, prime partner and type, title, budget (USD), dates submitted / approved / start / end, status, number of subpartners, total subgrants, subgrants to national NGOs.
- `uhf_subgrants_2019_2026.csv`: one row per subgrant. Project code, year, prime partner and type, subpartner name, subpartner type, amount (USD).
- `uhf_localization_by_year.csv`: per allocation year. Projects, prime partners, total budget, budget of national NGO prime partners (USD and %), subgrants to national NGOs from non-national primes, national NGO share including those subgrants, and national-to-national subgrants (reported separately).
- `uhf_budget_by_cluster_year.csv`: budget by cluster and allocation year.

## Method
- The source has one row per project × cluster × location. Project budgets are taken once per project; cluster budgets are summed from the atomic field BudgetByClusterByAdminLocation1 (the sum matches project budgets to the cent).
- Subgrant fields (SubIPName, SubIPAmt, SubIPTypeId) are split on '##'. Subpartner type codes: 3 = National NGO, 2 = International NGO, 5 = Other (faith-based organisations, institutions), 4 = Unknown.
- Localization 'including subgrants' = national NGO prime budgets + subgrants of type 3 received from international NGOs, UN agencies and other non-national primes. National-to-national subgrants are excluded from this share to avoid double counting and are reported in a separate column.
- Beneficiary counts from the source are not included: they are planned figures and repeat across cluster rows.
- Organisation names are kept as in the source (no normalisation).

## Validation (CONFIRMED)
For 2024 the dataset reproduces the UHF Annual Report 2024 (OCHA, 18.10.2025): total allocations USD 164.34 million (report: over USD 164 million), 58 partners (report: 58), USD 96.18 million or 58.5% to national NGOs including subgrants (report: USD 96 million, almost 60%).

## Key figures (CALCULATED)
National NGO share, direct / including subgrants: 2022 23.2% / 34.5%; 2023 20.8% / 32.8%; 2024 45.5% / 58.5%; 2025 58.0% / 69.0%; 2026 (partial) 41.3% / 53.3%.

## Licence
Compilation, cleaning and calculations: CC BY 4.0, Civic Intelligence. Underlying data: OCHA Country-Based Pooled Funds Data Hub (https://cbpf.data.unocha.org/); please also credit OCHA CBPF.

## How to cite
Civic Intelligence (2026). Ukraine Humanitarian Fund (UHF): projects, subgrants and localization, 2019–2026. Zenodo. DOI: see record.

Related analysis (Ukrainian): https://civicintelligence.digital/articles/ukrainskyi-humanitarnyi-fond-465-zaiavok.html
Method: https://civicintelligence.digital/metod/

---
## Українською
Повний перелік 465 проєктів Українського гуманітарного фонду (UHF, OCHA) за 2019–2026 роки, 747 субгрантів і річний показник локалізації: яка частка коштів доходить до національних НУО напряму і з урахуванням субгрантів. Цифри 2024 року звірено з річним звітом фонду. Ліцензія CC BY 4.0.
