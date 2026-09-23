# License status

The QSARify workspace root and the VPS project root both contain the same 1,077-byte `MIT License` file (`QSARify contributors`, SHA-256 `9b4cf68c9508e482317b83bf01c2a6bb7750f7e7da1864540381df1d21b60ee6`). On 2026-09-23, the user explicitly selected MIT for the Qsarify-Core candidate; that existing license text is included unchanged.

The user's MIT selection authorizes the license choice for this staging artifact. Separate coauthor/institutional redistribution authority has not yet been documented. The GitHub repository remains private until public-release readiness is confirmed.

## Options and recommendation

- **MIT (existing source-tree notice):** permissive commercial reuse and modification with preservation of the copyright and license notice. The [OSI MIT text](https://opensource.org/license/mit) confirms its notice condition and broad grant. It contains no detailed express patent license.
- **Apache-2.0:** also permissive and commercial-friendly, but adds express patent terms and redistribution conditions for notices/modified files. The [Apache Software Foundation text](https://www.apache.org/licenses/LICENSE-2.0) includes the patent grant, patent-litigation termination, and notice requirements.
- **GPL-3.0:** copyleft applies when covered software is conveyed/distributed; recipients get source rights under GPL terms. It does not add AGPL's separate network-interaction source-offer requirement. See the [GNU GPLv3 guide](https://www.gnu.org/licenses/quick-guide-gplv3.html).
- **AGPL-3.0:** adds a source-offer obligation for users interacting remotely with a modified network service. That is a meaningful constraint for hosted QSARify; see the [GNU AGPL overview](https://www.gnu.org/licenses/) and [GNU explanation](https://www.gnu.org/licenses/why-affero-gpl.html).

**Decision:** MIT is selected. It preserves the notice already present in both source roots and offers permissive reuse without imposing a copyleft obligation on QSARify's hosted service. Apache-2.0 was not selected; dual licensing was not requested.

An OSI license covers eligible software, not ChEMBL/PubMed-derived data, reviewer labels, paper artifacts, logos, or the QSARify trademark. Those remain separately rights-cleared or excluded. This is a release-planning recommendation, not legal advice. Obtain coauthor/institutional rights-holder approval before making the repository public or depositing the software in Zenodo.
