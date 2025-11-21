export interface BoundingBox {
  id: string | number;
  page: number;
  type: 'text' | 'table' | 'chart';
  coordinates: [number, number, number, number];
  label: string;
}

export interface ColumnDef {
  field: string;
  headerName: string;
}

export interface ExtractedData {
  columns: ColumnDef[];
  rows: any[];
  tables: any[];
  keyValuePairs: Record<string, string | number>;
  text: string;
  metadata?: any;
  errors: { row: number; column: string; reason: string }[];
}

export interface Insight {
  id: string;
  type: 'summary' | 'trend' | 'warning' | 'anomaly';
  content: string;
}

export type SummaryInsight = Insight;
