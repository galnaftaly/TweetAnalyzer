import React, { useState, Fragment } from 'react';
import { nanoid } from 'nanoid';
import ReadOnlyRowBook from './ReadOnlyRawBook';
import { styled } from '@mui/material/styles';
import ContentPasteGoIcon from '@mui/icons-material/ContentPasteGo';
import {
  TableContainer,
  Button,
  Table,
  TableBody,
  TableHead,
  TableRow,
  Box,
  Container,
  Paper,
  Alert,
  TextField,
  Typography,
  Select,
  MenuItem,
} from '@mui/material';
import AddBoxIcon from '@mui/icons-material/AddBox';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: theme.palette.common.blue,
    color: theme.palette.common.white,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}));
const bookOptions = [
  'A MIDSUMMER NIGHT_S DREAM',
  'KING HENRY VI Part III',
  'KING RICHARD III',
  'KING HENRY V',
  'THE MERRY WIVES OF WINDSOR',
  'Arden of Feversham',
  'The London Prodigal',
  'KING HENRY VI part I',
  'Comedy of Errors',
  'THE RAPE OF LUCRECE',
  'KING RICHARD II',
  'KING HENRY VIII',
  'AS YOU LIKE IT',
  'CYMBELINE',
  'KING HENRY VI Part II',
  'THE LIFE OF TIMON OF ATHENS',
  'MUCH ADO ABOUT NOTHING',
  'THE TRAGEDY OF CORIOLANUS',
  'THE TAMING OF THE SHREW',
  'THE TEMPEST',
  'THE TRAGEDY OF ANTONY AND CLEOPATRA',
  'THE TRAGEDY OF TITUS ANDRONICUS',
  'Venus and Adonis',
  'MEASURE FOR MEASURE',
  'THE WINTERS TALE',
  'Taming of the Shrew',
  'Merchant of Venice',
  'Love_s Labour_s Lost',
];

const BookTable = (props) => {
  const [alert, setAlert] = useState(false);

  const [addFormData, setAddFormData] = useState({
    text: '',
  });
  const [value, setValue] = useState('');

  const [editFormData, setEditFormData] = useState({
    text: '',
  });

  const [editBookId, setEditBookId] = useState(null);

  const handleAddFormChange = (event) => {
    event.preventDefault();
    setValue(event.target.value);
    const fieldName = 'text';
    const fieldValue = event.target.value;
    console.log(event.target.value);

    const newFormData = { ...addFormData };
    newFormData[fieldName] = fieldValue;

    setAddFormData(newFormData);
  };

  const handleAddFormSubmit = (event) => {
    event.preventDefault();
    setValue('');

    const newBook = {
      id: nanoid(),
      text: addFormData.text,
    };

    const newBooks = [...props.books, newBook];
    props.setBooks(newBooks);
    setAlert(false);
  };

  const handleEditFormSubmit = (event) => {
    event.preventDefault();

    const editedBook = {
      id: editBookId,
      text: editFormData.text,
    };

    const newBooks = [...props.books];

    const index = props.books.findIndex((book) => book.id === editBookId);

    newBooks[index] = editedBook;

    props.setBooks(newBooks);
    setEditBookId(null);
  };

  const handleDeleteClick = (bookId) => {
    const newBooks = [...props.books];
    const index = props.books.findIndex((book) => book.id === bookId);
    newBooks.splice(index, 1);
    props.setBooks(newBooks);
  };

  const sendingBooksTable = () => {
    if (props.books.length === 0) {
      setAlert(true);
    } else {
      props.setFetch(true);
      props.setTweetTable(props.books);
    }
  };

  return (
    <Box
      spacing={1}
      direction="column"
      align="center"
    >
      <form onSubmit={handleEditFormSubmit}>
        <TableContainer component={Paper} sx={{ maxHeight: 600, width: 1000 }}>
          <Table
            sx={{
              width: 'max-content',
              minWidth: 1000,
              maxWidth: 1000,
              borderCollapse: 'collapse',
            }}
            stickyHeader={true}
            aria-label="customized table"
          >
            <TableHead sx={{ display: 'table-header-group' }}>
              <TableRow>
                <StyledTableCell
                  align="left"
                  sx={{
                    fontSize: 'h5.fontSize',
                    border: 1,
                    borderColor: 'primary.main',
                  }}
                >
                  Shakespeare Books
                </StyledTableCell>
                <StyledTableCell
                  align="left"
                  width="230"
                  sx={{
                    border: 1,
                    borderColor: 'primary.main',
                    fontSize: 'h5.fontSize',
                  }}
                >
                  Actions
                </StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {props.books.map((book, index) => (
                <Fragment>
                  <ReadOnlyRowBook
                    key={`${index}`}
                    book={book}
                    handleDeleteClick={handleDeleteClick}
                  />
                </Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </form>

      <Container
        spacing={1}
        direction="column"
        justifyContent="center"
      >
        <Typography
          variant="h4"
          justifyContent="center"
          align="center"
          sx={{ m: 3 }}
        >
          Add a new Book:
        </Typography>
        <form onSubmit={handleAddFormSubmit}>
          <Select
          sx={{minWidth: 350}}
            autoFocus={true}
            margin="normal"
            variant="standard"
            type="text"
            name="text"
            required
            autoWidth={true}
            label="Select a Book from list..."
            onChange={handleAddFormChange}
            value={value}
          >
            {bookOptions.map((element, index) => (
              <MenuItem value={bookOptions[index]}>
                {bookOptions[index]}
              </MenuItem>
            ))}
          </Select>

          <Button endIcon={<AddBoxIcon />} type="submit">
            Add
          </Button>
          <Box>
            <Button
              size="large"
              variant="contained"
              color="success"
              onClick={sendingBooksTable}
              endIcon={<ContentPasteGoIcon />}
              sx={{ m: 2 }}
            >
              Analyze
            </Button>
            {alert && (
              <Alert severity="error">
                <Typography sx={{ fontSize: 16 }}>
                  You must insert text to analyze
                </Typography>
              </Alert>
            )}
          </Box>
        </form>
      </Container>
    </Box>
  );
};

export default BookTable;
