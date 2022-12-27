import React from "react";
import SaveIcon from "@mui/icons-material/Save";
import CancelIcon from "@mui/icons-material/Cancel";
import TableCell, { tableCellClasses } from "@mui/material/TableCell";
import { styled } from "@mui/material/styles";
import {
  Button,
  TableRow,
} from "@mui/material";

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: theme.palette.common.blue,
    color: theme.palette.common.white,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
    border:"1pt solid black"
  },
}));

const EditableRow = ({
  editFormData,
  handleEditFormChange,
  handleCancelClick,
}) => {
  return (
    <TableRow>
      <StyledTableCell >
        <input
          type="text"
          required="required"
          placeholder="Enter a tweet text..."
          name="tweetText"
          value={editFormData.tweetText}
          onChange={handleEditFormChange}
        ></input>
      </StyledTableCell>

      <StyledTableCell align="center">
        <Button type="submit" variant="contained" endIcon={<SaveIcon />}>
          Save
        </Button>
        <Button
          type="button"
          variant="contained"
          endIcon={<CancelIcon />}
          onClick={handleCancelClick}
        >
          Cancel
        </Button>
      </StyledTableCell>
    </TableRow>
  );
};

export default EditableRow;