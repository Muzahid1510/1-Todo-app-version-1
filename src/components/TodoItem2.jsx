function TodoItem2() {
  let TodoName = "Go to college";
  let TodoDate = "4/05/2024";
  return (
    <div className="container ">
      <div className="row kg-row">
        <div className="col-6">{TodoName}</div>
        <div className="col-4">{TodoDate}</div>
        <div className="col-2">
          <button type="button" className="btn btn-danger kg-button">
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

export default TodoItem2;
