# License status

The QSARify workspace root and the VPS project root both contain the same 1,077-byte `MIT License` file (`QSARify contributors`, SHA-256 `9b4cf68c9508e482317b83bf01c2a6bb7750f7e7da1864540381df1d21b60ee6`). This is evidence that the current source trees carry an MIT notice; the file alone does not establish that every author or institutional rights holder authorized relicensing or redistribution of every backend contribution.

The user supplied “Apache-2.0; MIT”. It remains unclear whether this confirms continuing under the existing MIT notice, selects between MIT and Apache-2.0, or requests dual licensing under `MIT OR Apache-2.0`. The backend-only candidate therefore does not yet include a `LICENSE` file. The GitHub repository is private and empty; no code has been pushed.

## Options and recommendation

- **MIT (existing source-tree notice):** permissive commercial reuse and modification with preservation of the copyright and license notice. The [OSI MIT text](https://opensource.org/license/mit) confirms its notice condition and broad grant. It contains no detailed express patent license.
- **Apache-2.0:** also permissive and commercial-friendly, but adds express patent terms and redistribution conditions for notices/modified files. The [Apache Software Foundation text](https://www.apache.org/licenses/LICENSE-2.0) includes the patent grant, patent-litigation termination, and notice requirements.
- **GPL-3.0:** copyleft applies when covered software is conveyed/distributed; recipients get source rights under GPL terms. It does not add AGPL's separate network-interaction source-offer requirement. See the [GNU GPLv3 guide](https://www.gnu.org/licenses/quick-guide-gplv3.html).
- **AGPL-3.0:** adds a source-offer obligation for users interacting remotely with a modified network service. That is a meaningful constraint for hosted QSARify; see the [GNU AGPL overview](https://www.gnu.org/licenses/) and [GNU explanation](https://www.gnu.org/licenses/why-affero-gpl.html).

**Recommendation:** retain the existing MIT license for the backend release, if the other authors/institution confirm they can authorize the included contributions. This preserves the license already present in both source roots and matches a permissive research-software release without imposing a copyleft obligation on QSARify's hosted service. Choose Apache-2.0 instead if the authors specifically want its express patent grant. Dual `MIT OR Apache-2.0` gives downstream users a choice but adds no necessary benefit for this application unless that choice is an explicit project goal.

An OSI license covers eligible software, not ChEMBL/PubMed-derived data, reviewer labels, paper artifacts, logos, or the QSARify trademark. Those remain separately rights-cleared or excluded. This is a release-planning recommendation, not legal advice. Confirm the user's intended scope and obtain the other authors'/institutional rights-holder approval before making a public release or changing the existing license terms.
