import { BoundingBox, ExtractedData, Insight } from './appTypes';

export const MOCK_BOUNDING_BOXES: BoundingBox[] = [
    {
        id: '1',
        page: 1,
        type: 'text',
        coordinates: [0.1, 0.1, 0.9, 0.2],
        label: 'Sample Header Text'
    },
    {
        id: '2',
        page: 1,
        type: 'table',
        coordinates: [0.1, 0.3, 0.9, 0.6],
        label: 'Financial Table'
    },
    {
        id: '3',
        page: 1,
        type: 'chart',
        coordinates: [0.1, 0.65, 0.5, 0.9],
        label: 'Revenue Chart'
    }
];

export const MOCK_EXTRACTED_DATA: ExtractedData = {
    columns: [
        { field: 'id', headerName: 'ID' },
        { field: 'date', headerName: 'Date' },
        { field: 'amount', headerName: 'Amount' },
        { field: 'status', headerName: 'Status' }
    ],
    rows: [
        { id: 1, date: '2023-01-01', amount: 1000, status: 'Completed' },
        { id: 2, date: '2023-01-02', amount: 1500, status: 'Pending' },
        { id: 3, date: '2023-01-03', amount: 2000, status: 'Completed' },
        { id: 4, date: '2023-01-04', amount: 1200, status: 'Failed' }
    ],
    tables: [],
    keyValuePairs: {
        'Total Revenue': '$5,700',
        'Period': 'Q1 2023',
        'Report Date': '2023-04-15'
    },
    text: 'This is a sample extracted text from the document. It contains financial data for Q1 2023.',
    errors: []
};

export const MOCK_INSIGHTS: Insight[] = [
    {
        id: 'summary-1',
        type: 'summary',
        content: 'The document contains financial data for Q1 2023, showing a total revenue of $5,700.'
    },
    {
        id: 'trend-1',
        type: 'trend',
        content: 'Revenue shows a positive trend in the first three days.'
    },
    {
        id: 'warning-1',
        type: 'warning',
        content: 'One transaction on Jan 4th failed.'
    }
];
