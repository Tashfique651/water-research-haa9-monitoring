# Source data and attribution

Publisher data, documentation and literature retain their original terms. This repository includes processed observations, derived statistics and selected government documentation, not the complete publisher archives. The source versions and hashes below identify the historical inputs; no new project license is supplied.

## U.S. Environmental Protection Agency sources

| Source | Recorded coverage | Publisher URL | Frozen SHA-256 | Manifest description |
|---|---|---|---|---|
| UCMR 4 occurrence data | 2018-2020, March 2024 revision | https://www.epa.gov/system/files/other-files/2024-04/ucmr4-occurrence-data.zip | `932dd46139c1aef050d020054c546fea4178b97bcf12a4e455c050074d17b255` | U.S. federal public data |
| Fourth Six-Year Review HAA5 | 2012-2019 | https://www.epa.gov/system/files/other-files/2024-03/syr4_haas.zip | `8bc74a7ad488928454f96681db933d7c7bf3ab894b9e274a9e9fe4b511ed9926` | U.S. federal public data |
| Fourth Six-Year Review TTHM | 2012-2019 | https://www.epa.gov/system/files/other-files/2024-03/syr4_thms.zip | `3b2a8a44b27a1e94e053d14c350fd7fad7d256d554ed2818de3317e06af81068` | U.S. federal public data |
| Fourth Six-Year Review DBP-related parameters | 2012-2019 | https://www.epa.gov/system/files/other-files/2024-03/syr4_dbp_related-parameters.zip | `565fbb6e6fa3befe312ea4b2aec74e93cdadd4eb33a4f690762ad3678e639615` | U.S. federal public data |
| Fourth Six-Year Review disinfectant residuals | 2012-2019 | https://www.epa.gov/system/files/other-files/2024-03/syr4_disinfectant-residuals.zip | `ef8545b0799cf2a72c883c8564919044dc4ec1a101b190507a16b8e0a248bcb2` | U.S. federal public data |
| Fourth Six-Year Review treatment and water-system tables | 2012-2019 | https://www.epa.gov/system/files/other-files/2024-03/syr4_treatment.zip | `37300249a7b2201ac5d143b99f5ff3984933b7ab48aa6234f6a9fceed239eeb3` | U.S. federal public data |
| Information Collection Rule auxiliary database | 1997-1998 | https://www.epa.gov/sites/default/files/2017-07/dbpicr_aux1_updated.zip | `d91454d34d535f8ce36905dfad4d9c9f40b3a64f912be1a3c7edcc62abf09328` | U.S. federal public data |
| Third Six-Year Review HAA records | 2006-2011 | https://www.epa.gov/sites/default/files/2016-12/syr3_haa.zip | `259c8fcf20a321fed68d6124768b33561d1c3075d68de09a5327140e02a5c19f` | U.S. federal public data |
| Third Six-Year Review THM records | 2006-2011 | https://www.epa.gov/sites/default/files/2016-12/syr3_thm.zip | `8ba57f6e1309c7467013d37f822f3c3b7d2ec442c96c6b6955d26a14be9d1a40` | U.S. federal public data |
| Third Six-Year Review data user guide | December 2016 | https://www.epa.gov/sites/default/files/2016-12/documents/user_guide_to_obtaining_and_using_syr3_data.pdf | `9ee669baeb751cc7aad85a038870b0813deb7f90003edd91fe367a5faf38619b` | U.S. federal public document |
| Third Six-Year Review data management and QA/QC report | EPA-810-R-16-015, December 2016 | https://www.epa.gov/sites/default/files/2016-12/documents/810r16015_0.pdf | `78bb248489175f286049908ee171ecabc5812a883db39f47e6172ea75d22b2c5` | U.S. federal public document |

## California sources

| Source | Recorded version | Publisher URL | Frozen SHA-256 | Manifest description |
|---|---|---|---|---|
| California CLIP monitoring extract | 2025 frozen CSV | https://data.ca.gov/dataset/d7193835-8161-4ff6-bcc3-da894965d53c/resource/84de0856-5b00-421f-83de-bd78d57c4b08/download/clip-sql-2025.csv | `1d924cd3b28b1fee0fd861d23750069391e36fda8a23d9d5c73ca63b4dd4ea70` | California Open Data, CC BY attribution stated on the portal |
| California SDWIS4 water-quality archive | Frozen download used for the 2023-2025 analysis | https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/documents/edtlibrary/SDWIS4.zip | `a4d8543490ff9d42ca180fcb5851c8d1e44793647531188350ee9e6fff2e1766` | California State Water Resources Control Board public monitoring archive; verify current publisher terms before redistribution |
| California EDT Library data dictionary | Retrieved 2026-09-13 | https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/documents/edtlibrary/data_dictionary.pdf | `6e0901876f98bafc67495dc13afe84d7b4ca9f6337537d2c058283257cea53e0` | California public document |

The CLIP digest identifies the original download. The historical encoding transformation is documented in `data/manifests/CLIP_packaged_encoding_provenance.json`; the raw CSV is not included here.

The SDWIS4 archive and its 2.19 GB extracted member are not included. `src/wr_acquire_california.py` documents acquisition, digest verification and extraction. Recheck the publisher terms before redistributing the archive.

## Species-level repository

The frozen Peterson et al. HAA species repository archive was acquired from
https://github.com/wraseman/haa-cytotoxicity-occurrence at commit
`e0dd19e8f392ac08aa3a00e6c09622014d074e7c`. The project manifest records the
repository distribution as MIT licensed and the underlying EPA records as U.S.
federal public data. The frozen repository archive has SHA-256 digest
`565c5ee9d695aabc7387f2754f6698edbb26e41b4b33bffb50704cd11285127f`.

## Additional UK evidence and laboratory-cost sources

The September 2026 search is documented in wr_results/haa9_revision/external_acquisition_ledger.json and CSV, with successful, blocked and analytically insufficient outcomes. This ledger distinguishes collection periods from release dates and original downloads from text extraction or digitization.

- The Cranfield dataset by P. Grundy, E. H. Goslan, P. Jarvis and B. Jefferson is available at https://doi.org/10.57996/cran.ceres-2807 under Creative Commons Attribution 4.0 International. Attribute its creators and publisher, link the license at https://creativecommons.org/licenses/by/4.0/, and identify changes. The retained README and CSV are source-reader extracts, not original-byte files. A micro sign in CSV headers was altered during extraction; numerical cells are retained unchanged and units were confirmed from the README. Derived files compute row-wise group-mean sums, not individual sample results. Its 30 group rows overlap and are not a longitudinal validation dataset.
- WRc report UC8493, DWI70/2/253, by M. Harman, P. Rumsby and R. Kanda (May 2011), was downloaded from the official Drinking Water Inspectorate archive. The report expressly reserves copyright. Its full PDF and rendered pages are internal verification material only and excluded from releases. The attributed CSV digitizes numerical facts from Tables 6.2-6.5, retaining source dates, nondetect symbols and rounding. It is not an original laboratory export. No project-code license applies to it.
- Weck's 2007 fee schedule in a Los Angeles municipal packet and the Pace quotation in the January 2018 North Shore Water Commission packet were consulted for price facts. Complete packets and screenshots are excluded from public packages; only attributed numeric facts, source locations, limitations and reference metadata are retained.
- SimpleLab HAA5 and HAA9 product pages were read on 19 September 2026. Their price facts are vendor-package illustrations with different listed methods and do not establish a current matched-laboratory analytical price ratio. No complete commercial page is reproduced.
- EPA Methods 552.3 and 557 are retained as public U.S. government methodological documents. Their holding-time provisions concern distinct extraction and analysis workflows and must not be collapsed into a universal holding time.

Never include wr_results/haa9_revision/source_page_checks/ in a Git release or submission ZIP. It contains restricted source PDFs and page images, not original publication figures. The current release policy rejects that path. Neither public accessibility nor inclusion in a local working directory grants redistribution rights.

## Derived outputs

Derived tables in a release package should include a provenance manifest that
records the source archive digest, generating script, software environment, row
count, and output digest. Do not infer permission to redistribute a source file
from permission to redistribute code or derived statistics.
