rules = {
    "GROUP BY": "Focus on the column(s) that define the groups. If the two SQLs group by different columns, analyze which group is more appropriate according to the NL query and evidence (if any).",
    "ORDER BY": "If the SQLs contain an ORDER BY clause, compare the ordering columns and directions. Ensure they meet the NL Query's requirements. If Order By a numeric column in an ascending order, check if the numeric column has null values.",
}
