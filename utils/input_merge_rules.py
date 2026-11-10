"""Explicit normalization for reviewed input-table consolidations."""
from utils.data_store import describe, quote

PROFILES = {'forecast', 'fund', 'price', 'shareholder', 'restructuring', 'security'}
GROUP = ['stock_code', 'pub_date', 'end_date', 'holder_scope']
SHARED_HOLDER = GROUP + ['share_ratio', 'shareholder_class_id']


def normalize(conn, relation, profile, *, source=False, resolutions=None):
    cols = dict(describe(conn, relation))
    expressions = {k: quote(k) for k in cols}
    if profile == 'security':
        if not resolutions or len({r['stock_code'] for r in resolutions}) != len(resolutions):
            raise ValueError('Security names require distinct, evidence-bound resolutions')
        literal = lambda value: "'" + str(value).replace("'", "''") + "'"
        for row in resolutions:
            if not row.get('checked_at') or not row.get('evidence_urls'):
                raise ValueError('Security name resolution lacks dated evidence')
            accepted = row['accepted_names'] + [[row['display_name'],row['name']]]
            marks = ','.join('(?,?)' for _ in accepted)
            params = [row['stock_code']] + [v for pair in accepted for v in pair]
            if conn.execute(f'SELECT 1 FROM {relation} s WHERE stock_code=? AND '
                            f'NOT EXISTS (SELECT 1 FROM (VALUES {marks}) n(display_name,name) '
                            'WHERE n.display_name IS NOT DISTINCT FROM s.display_name AND n.name IS NOT DISTINCT FROM s.name) LIMIT 1',params).fetchone():
                raise ValueError('Security name changed beyond reviewed snapshots; verify fresh evidence: '+row['stock_code'])
        for field in ('display_name','name'):
            expressions[field] = 'CASE stock_code ' + ' '.join('WHEN '+literal(row['stock_code'])+' THEN '+literal(row[field]) for row in resolutions) + ' ELSE '+quote(field)+' END'
    if profile == 'fund':
        code = quote('stock_code')
        expressions['stock_code'] = (f"CASE WHEN regexp_full_match({code}, '[036][0-9]{{5}}') "
            f"THEN {code} || CASE WHEN starts_with({code}, '6') THEN '.XSHG' ELSE '.XSHE' END "
            f"WHEN regexp_full_match({code}, '[036][0-9]{{5}}\\.(XSHG|XSHE)') THEN {code} "
            "ELSE error('Unrecognized stock code in fund holdings') END")
    if profile == 'restructuring' and source:
        if conn.execute(f'SELECT 1 FROM {relation} WHERE event_date IS DISTINCT FROM availability_date '
                        "OR source_key IS DISTINCT FROM source_table || ':' || json_extract_string(details_json, '$.ID') LIMIT 1").fetchone():
            raise ValueError('Restructuring source identity/disclosure date mismatch')
        expressions['source_id'] = "json_extract_string(details_json, '$.ID')"
        expressions['pub_date'] = expressions.pop('availability_date')
        expressions['availability_date'] = expressions['event_date']
        date = lambda field: f"CAST(json_extract_string(details_json, '$.{field}') AS TIMESTAMP)::DATE"
        expressions['event_date'] = ("CASE source_table WHEN 'jy.LC_Merger' THEN "
            f"coalesce({date('AgreementDate')},{date('EndDate')}) WHEN 'jy.LC_AssetBuyAndSale' THEN "
            f"coalesce({date('TradingDay')},{date('EndDate')}) WHEN 'jy.LC_PurchaseAndSale' THEN {date('EndDate')} "
            "ELSE error('Unknown restructuring provider table') END")
        expressions['event_type'] = ("CASE source_table WHEN 'jy.LC_Merger' THEN coalesce(json_extract_string(details_json,'$.ChangeType'),'merger') "
            "WHEN 'jy.LC_AssetBuyAndSale' THEN json_extract_string(details_json,'$.EventType') "
            "ELSE json_extract_string(details_json,'$.PurchaseType') END")
        expressions['amount'] = ("CASE source_table WHEN 'jy.LC_Merger' THEN coalesce(json_extract_string(details_json,'$.FinalAmount'),json_extract_string(details_json,'$.InitialAmount')) "
            "WHEN 'jy.LC_AssetBuyAndSale' THEN json_extract_string(details_json,'$.Price') "
            "ELSE json_extract_string(details_json,'$.TradeSum') END")
    if profile == 'shareholder':
        if source:
            expressions['end_date'] = expressions.pop('report_date')
            expressions['holder_scope'] = "'total'::VARCHAR"
        if cols['shareholder_class_id'] != 'INTEGER':
            if conn.execute(f'SELECT 1 FROM {relation} WHERE shareholder_class_id IS NOT NULL AND '
                            'CAST(shareholder_class_id AS DOUBLE) IS DISTINCT FROM '
                            'CAST(CAST(shareholder_class_id AS INTEGER) AS DOUBLE) LIMIT 1').fetchone():
                raise ValueError('Lossy integer conversion: shareholder_class_id')
        expressions['shareholder_class_id'] = 'CAST(shareholder_class_id AS INTEGER)'
        # Identity is local to this carrier; never guess a provider ID for a named holder.
        if '_input_row_id' not in cols:
            if source:
                expressions['_input_row_id'] = "'provider:' || CAST(source_id AS VARCHAR)"
            else:
                fields = ','.join(f'{quote(k)}:={quote(k)}' for k in cols)
                ordered = ','.join(quote(k) for k in cols)
                expressions['_input_row_id'] = f"'detail:' || md5(to_json(struct_pack({fields}))) || ':' || row_number() OVER(PARTITION BY {ordered})::VARCHAR"
    sql = 'SELECT ' + ','.join(v+' AS '+quote(k) for k,v in expressions.items()) + ' FROM '+relation
    return sql, dict(describe(conn, '('+sql+')'))


def holder_union(conn, a, b, schema):
    """Keep complete ordinary batches; accept only disjoint or equal thin batches."""
    group = ' AND '.join(f'a.{quote(k)} IS NOT DISTINCT FROM b.{quote(k)}' for k in GROUP)
    fields = ','.join(map(quote,SHARED_HOLDER))
    # INTERSECT/EXCEPT ALL is essential: equal ratios may belong to multiple holders.
    for left,right in ((a,b),(b,a)):
        sql = (f'WITH a AS ({left}),b AS ({right}) SELECT 1 FROM '
               f'(SELECT {fields} FROM a WHERE EXISTS (SELECT 1 FROM b WHERE {group}) '
               f'EXCEPT ALL SELECT {fields} FROM b WHERE EXISTS (SELECT 1 FROM a WHERE {group})) LIMIT 1')
        if conn.execute(sql).fetchone():
            raise ValueError('Shareholder disclosure batch differs; explicit provider revision review required')
    return f'WITH a AS ({a}),b AS ({b}) SELECT * FROM a UNION ALL SELECT * FROM b WHERE NOT EXISTS (SELECT 1 FROM a WHERE {group})'


def text_extension(a, b):
    # Only the reviewed carriage-return truncation is repairable without a revision decision.
    return f"starts_with(replace({b}, chr(13), ''), replace({a}, chr(13), ''))"
