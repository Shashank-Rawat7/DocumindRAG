# Evaluation corpus: sources and licences

This file accompanies `manifest.json` in this folder. `manifest.json` and this file are metadata,
not corpus documents. The six corpus documents are listed in section 1 and in `manifest.json`.
Do not distribute any corpus file without this file.

The stored files are copies of the upstream files, with one exception: the two PostgreSQL files
are page-range extracts of a larger PDF (section 2.5). Nothing else was edited. The manifest has
no per-document file-name or page-range field, so the key-to-file mapping and the page ranges are
recorded here. The SHA-256 of every stored file is in `manifest.json` (`source_file_sha256`).

## 1. Corpus documents and the terms that apply

| Corpus key | Stored file | Upstream source (pinned) | Terms for the documentation text | Terms for code examples in the text |
|---|---|---|---|---|
| docker-build-best-practices | docker-build-best-practices.md | docker/docs, commit 61cbfdc1e72d0ac62dbac59ceaf7b94dcbe69c80 | Apache-2.0 (section 3.3) | Apache-2.0 (same repository licence) |
| docker-compose-startup-order | docker-compose-startup-order.md | docker/docs, same commit | Apache-2.0 (section 3.3) | Apache-2.0 (same repository licence) |
| fastapi-deployment-docker-en | fastapi-deployment-docker-en.md | fastapi/fastapi, tag 0.162.5, commit 5add7a458d629b07311d48261f696b5ac2bb9a5f | MIT (section 3.2) | MIT (same repository licence) |
| pg18-concurrency | postgresql-18.6-concurrency-control.pdf | PostgreSQL 18.6 US manual, Chapter 13 | PostgreSQL License (section 3.1) | PostgreSQL License (same notice) |
| pg18-indexes | postgresql-18.6-indexes.pdf | PostgreSQL 18.6 US manual, Chapter 11 | PostgreSQL License (section 3.1) | PostgreSQL License (same notice) |
| python314-asyncio-dev | python314-asyncio-dev.txt | python/cpython, tag v3.14.0, commit ebf955df7a89ed0c7968f79faec1de49f61ed7cb | PSF License Version 2 (section 3.4) | PSF License Version 2 and Zero-Clause BSD (sections 3.4 and 3.5) |

Images and other assets referenced by a document are not part of this corpus (see section 2).

Third-party content review: the six stored documents contain no copyright or licence statements of
their own. Images and other assets that a document refers to (one Docker image and two FastAPI
screenshots, listed in section 2) are excluded from this corpus and are NOT redistributed here. Their
licensing was not examined, and nothing in this file clears them for redistribution. The licences below
cover only the stored documents.

## 2. Per-document provenance

### 2.1 docker-build-best-practices
- Source file: `content/manuals/build/building/best-practices.md` in `docker/docs` at the commit above.
- Stored unchanged. SHA-256 `349577b488b7867670ac6264908303ba748bcc81682bdf33ef802d4e2e35ad6e`.
- Licence basis: the repository `README.md` at that commit states "Copyright 2013-2026 Docker, Inc.,
  released under the Apache 2.0 license", and the repository root `LICENSE` is the Apache License,
  Version 2.0. No separate licence for the documentation content was found in the repository. There is no
  `NOTICE` file at that commit.
- Excluded asset: the page refers to the image `../images/hub-official-images.webp`. The image is NOT
  part of this corpus; only its alt text and the relative reference remain in the Markdown.
- The text contains Dockerfile and shell examples and links to third-party sites. It contains no
  copyright or licence statement of its own.

### 2.2 docker-compose-startup-order
- Source file: `content/manuals/compose/how-tos/startup-order.md` in `docker/docs` at the same commit.
- Stored unchanged. SHA-256 `167fec8c700792a3ee72e45dd7232df8dc11c987853245a0af7afdbc2bfedcf7`.
- Licence basis: as in 2.1. Contains a YAML example. No image, no external link, no copyright or licence
  statement of its own.

### 2.3 fastapi-deployment-docker-en
- Source file: `docs/en/docs/deployment/docker.md` in `fastapi/fastapi` at the commit above. At that
  commit `fastapi/__init__.py` has `__version__ = "0.162.5"` and the tag `0.162.5` resolves to the commit.
- Stored unchanged. SHA-256 `0337b98d51778755f59aa1b96ba319f0decc20136f3b3ff1b7547619a5c71100`.
- Licence basis: the repository `LICENSE` at that commit is the MIT License with
  "Copyright (c) 2018 Sebastián Ramírez"; `pyproject.toml` declares `license = "MIT"`; the `README.md`
  states "This project is licensed under the terms of the MIT license."
- Excluded assets: the page refers to two images by absolute URL,
  `https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png` and
  `https://fastapi.tiangolo.com/img/index/index-02-redoc-simple.png`. They are screenshots of third-party
  tools. The images are NOT part of this corpus; only the Markdown reference and alt text remain.
  The licensing of those screenshots was not examined.
- The text contains Dockerfile, console, Python and JSON examples. No copyright or licence statement of
  its own.

### 2.4 python314-asyncio-dev
- Source file: `Doc/library/asyncio-dev.rst` in `python/cpython` at the commit above (annotated tag
  `v3.14.0` resolves to it).
- Source format: reStructuredText. It is stored unchanged as raw text with a `.txt` extension.
  SHA-256 `41975dccd5de917d9da288998a94de9e07a2a66f575ecaba07bde948541617e3`.
- Licence basis: https://docs.python.org/3.14/license.html states that "Python software and documentation
  are licensed under the Python Software Foundation License Version 2" and that "Starting with Python
  3.8.6, examples, recipes, and other code in the documentation are dual licensed under the PSF License
  Version 2 and the Zero-Clause BSD license." The documentation's configured copyright is "2001 Python
  Software Foundation" (`Doc/conf.py` at the pinned commit).
- The PSF licence (section 3.4) requires its text and the notice "Copyright (c) 2001 Python Software
  Foundation; All Rights Reserved" to be retained in copies; both are reproduced below. The file is a
  verbatim copy, so no summary of changes applies. Code examples in the text (reStructuredText literal
  blocks) are also available under Zero-Clause BSD (section 3.5).
- No images, includes or copyright statements of its own.

### 2.5 pg18-indexes and pg18-concurrency (PostgreSQL page-range extracts)
- Source PDF: https://www.postgresql.org/files/documentation/pdf/18/postgresql-18-US.pdf. This URL is
  not version-pinned. Version used: PostgreSQL 18.6; 3,329 pages;
  SHA-256 `625d8080532051a4e54ee53bc54eeb6a9fa5ef6fc8053a848101a901c996198b`.
- Whole pages were copied with pypdf 6.20.0 (`PdfReader`/`PdfWriter`, link annotations excluded,
  nothing re-rendered). Repeating the extraction in a second process gave identical files.

| Key | Stored file | Chapter | Printed pages | Zero-based PDF page indices (inclusive) | Pages | Size | SHA-256 |
|---|---|---|---|---|---|---|---|
| pg18-indexes | postgresql-18.6-indexes.pdf | 11. Indexes | 482-498 | 522-538 | 17 | 60,928 bytes | c6d6e64faba4d8eed1fb65676a7c1b0d65db157c15e8f1e02b0d16d7bebc023c |
| pg18-concurrency | postgresql-18.6-concurrency-control.pdf | 13. Concurrency Control | 543-558 | 583-598 | 16 | 63,655 bytes | c049b6a747df41859b9ced0095ac4d32533fe8f970f8ba60756e88e4a584f465 |

- The extracts contain no images, annotations, URLs, or copyright or licence statements. The PostgreSQL
  legal notice (section 3.1) is deliberately not part of the extracted pages, so it never enters the
  indexed text. It applies to both extracts and must accompany them.
- The retrieval date of the source PDF is not recorded.

## 3. Licence texts and notices

### 3.1 PostgreSQL legal notice (documentation of PostgreSQL 18.6)

Source: `doc/src/sgml/legal.sgml` at tag `REL_18_6` of https://github.com/postgres/postgres
(commit 724edf9bde9d356724ad384a2e196edc3c9f80f7), SHA-256 `75e566020e07cd7652a16ca7965234c3614fb9b023ff2d94f20f40a719f27cdb`. This is the source of the Legal
Notice printed in the 18.6 manual. Entity markup (`&copy;`, `<quote>`) is rendered and line wrapping
is normalised; the wording is unchanged. Applies to: `pg18-indexes` and `pg18-concurrency`
(documentation text and any examples in it).

> PostgreSQL Database Management System (also known as Postgres, formerly known as Postgres95)
>
> Portions Copyright © 1996-2026, PostgreSQL Global Development Group
>
> Portions Copyright © 1994, The Regents of the University of California
>
> Permission to use, copy, modify, and distribute this software and its documentation for any purpose, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and this paragraph and the following two paragraphs appear in all copies.
>
> IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS-IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

### 3.2 MIT License (FastAPI)

Source: `LICENSE` at commit 5add7a458d629b07311d48261f696b5ac2bb9a5f of https://github.com/fastapi/fastapi,
SHA-256 `4ec89ffc81485b97fec584b2d4a961032eeffe834453894fd9c1274906cc744e`. Reproduced verbatim. Applies to: `fastapi-deployment-docker-en` (documentation
text and code examples).

```text
The MIT License (MIT)

Copyright (c) 2018 Sebastián Ramírez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### 3.3 Apache License, Version 2.0 (Docker documentation)

Source: `LICENSE` at commit 61cbfdc1e72d0ac62dbac59ceaf7b94dcbe69c80 of https://github.com/docker/docs,
SHA-256 `089009c932d3cc296355e429fa9d6faf80ac89597da866e2424b4ea09add64b1`. Reproduced verbatim. Applies to: `docker-build-best-practices` and
`docker-compose-startup-order` (documentation text and code examples). Assets excluded in section 2 are not
covered by this file. Copyright statement at that commit (`README.md`): "Copyright 2013-2026 Docker, Inc.,
released under the Apache 2.0 license".

```text
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2016 Docker, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### 3.4 Python Software Foundation License Version 2 (Python documentation text)

Source: the section "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" of the `LICENSE` file at commit
ebf955df7a89ed0c7968f79faec1de49f61ed7cb of https://github.com/python/cpython, SHA-256 `b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231`.
Reproduced verbatim. Applies to: `python314-asyncio-dev` (the documentation text, and also the code
examples, which are available under both this licence and section 3.5). The other agreements in that
`LICENSE` file (BeOpen.com, CNRI, CWI) are historical and do not apply to this document.

```text
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001 Python Software Foundation; All Rights Reserved"
are retained in Python alone or in any derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
```

### 3.5 Zero-Clause BSD License (code examples in the Python documentation)

Source: the section "ZERO-CLAUSE BSD LICENSE FOR CODE IN THE PYTHON DOCUMENTATION" of the same `LICENSE`
file. Reproduced verbatim. Applies to: only the examples, recipes and other code in
`python314-asyncio-dev`, not its prose.

```text
ZERO-CLAUSE BSD LICENSE FOR CODE IN THE PYTHON DOCUMENTATION
----------------------------------------------------------------------

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## 4. Excluded candidates (not part of this corpus)

- FastAPI Hindi Docker page: translation rights and provenance unresolved; not used.
- Python documentation Hindi translation and Hindi Wikipedia: dropped during selection.
