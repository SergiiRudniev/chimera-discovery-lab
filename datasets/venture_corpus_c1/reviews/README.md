# Independent reviews

1. Open every registered SEC filing in `reviewer_packet.json`.
2. Copy `review_template.json` to `<review-id>.review.json` in this directory.
3. Fill the reviewer identity, attestations and every decision, including all seven
   human-assigned ratings on every node.
4. Run `chimera validate-review-gate`.

The reviewer must not be `chimera-corpus-author-001` and must not inspect model
or candidate outputs. `verified` is the only accepted item-level decision.
`needs_change` and `cannot_verify` keep generation blocked.
